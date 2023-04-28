import torch
from torch import nn
from CNN import CNN
from torch.utils.data import DataLoader, Dataset
from torch.nn.modules import CrossEntropyLoss
import torch.nn.functional as F
import torchvision
from scipy import stats

class client():
    def __init__(self, id):
        self.client_id = id
        self.dataset = torchvision.datasets.CIFAR10("./resource/cifar10", train=True,
                                            transform=torchvision.transforms.ToTensor(), download=True)
        self.local_module = self.__to_cuda__(CNN())
        #self.next = next
        self.__loss_fn = self.__to_cuda__(CrossEntropyLoss())
        self.learning_rate = 0.01
        self.__optim = torch.optim.SGD(self.local_module.parameters(), lr=self.learning_rate)
        self.batchsize = 2000
        self.next = None



    def __to_cuda__(self, module):
        if(torch.cuda.is_available()):
            return module.cuda()
        
    def set_parameters(self, parameters):
        self.local_module.load_state_dict(parameters)

    def get_parameters(self):
        return self.local_module.state_dict()

    def calculate_kl_div(self):
        self.dataloader = DataLoader(self.dataset, self.batchsize, True)
        self.train_data_iter = self.dataloader.__iter__().__next__()
        elem_num_list = torch.unique(self.train_data_iter[1], return_counts=True)[1]
        elem_sum = elem_num_list.sum()
        elem_p = elem_num_list.float() / elem_sum
        q = [1/10 for i in range(10)]
        self.kl_div = stats.entropy(elem_p.tolist(), q)
        return self.kl_div
        
    def set_next(self, next):
        self.next = next


    def update_parameters(self):
        self.local_module.eval()
        print("参与方{}开始训练..".format(self.client_id))
        image = self.__to_cuda__(self.train_data_iter[0])
        label = self.__to_cuda__(self.train_data_iter[1])
        #初始化梯度参数
        self.__optim.zero_grad()
        #计算输出
        output = self.local_module(image)
        #计算损失
        curr_loss = self.__loss_fn(output, label)
        curr_loss = curr_loss.requires_grad_()
        print("第{}个参与方迭代, 损失值：{}".format(self.client_id, curr_loss))
        #反向传播
        curr_loss.backward()
        #更新梯度
        self.__optim.step()
        return curr_loss
    
    def test(self, module_path=None):
        mini_batch = 200
        #如果提供了保存模型路径应该从路径读取
        if module_path != None:
            self.local_module.load_state_dict(torch.load(module_path))
        dataset = torchvision.datasets.CIFAR10("./resource/cifar10", train=False,
                                            transform=torchvision.transforms.ToTensor(), download=True)
        data_batch = DataLoader(dataset, mini_batch, True)

        with torch.no_grad():
            total_loss = 0
            for image, label in data_batch:
                image = self.__to_cuda__(image)
                label = self.__to_cuda__(label)
                output = self.local_module(image)
                
                loss_list = output.argmax(1)
                loss_list = (loss_list == label).sum()
                total_loss += loss_list.float()
            mean_loss = total_loss / (len(data_batch) * mini_batch)
            print("平均正确率为{}".format(mean_loss))
            return mean_loss

