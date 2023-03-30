from Module import cifar10
from torch.utils.data import DataLoader, Dataset
from torch.nn.modules import CrossEntropyLoss
from transformer import Transformer
import torch
import torchvision
from seal import *


class Client():
    def __init__(self, id, sealer):
        self.dataset = torchvision.datasets.CIFAR10("./resource/cifar10", train=True,
                                                    transform=torchvision.transforms.ToTensor(), download=True)
        self.local_module = self.__to_cuda__(cifar10())
        self.trans = Transformer()
        self.module_length = len(self.trans.para_to_list(self.local_module.state_dict(), self.local_module))
        self.sealer = sealer
        self.client_id = id
        self.learning_rate = 0.01
        self.pg = 5.0e-4
        self.ng = -5.0e-4

    def __to_cuda__(self, module):
        if(torch.cuda.is_available()):
            return module.cuda()
        

    def set_parameters(self, para, divide_num):
        '''设置参与方本地模型参数,'''
        if isinstance(para, cipher_list) == False:
            raise Exception("模型参数类型不正确.")
        #先解密
        parameters_list = self.sealer.decrypt(para, self.module_length)
        #再转化为tensor
        parameters_tensor = torch.tensor(parameters_list)
        #计算平均
        parameters_tensor = parameters_tensor / divide_num
        #转化为dict
        parameters_dict = self.trans.list_to_para(parameters_tensor, self.local_module)
        #设置参数
        self.local_module.load_state_dict(parameters_dict, strict=True)
        


    def set_gradient_rate(self, rate):
        self.pg = self.pg * rate
        self.ng = self.ng * rate

    def set_grad(self, parameters):
        with torch.set_grad_enabled(True):
            trans = Transformer()
            #迭代parameters的每个tensor
            for value in parameters:
                #获得每个tensor的grad
                ten_grad_peer_para = value.grad.data
                #转换为ng and pg
                big_bool = (ten_grad_peer_para > 0).float()
                small_bool = (ten_grad_peer_para < 0).float()
                big_num = big_bool * self.pg
                small_num = small_bool * self.ng
                ten_grad_peer_para = big_num + small_num
                value.grad.data = ten_grad_peer_para
            return parameters

    def update_parameters(self, epoch, mini_batch):
        #先设置模型参数
        #self.set_parameters(para)

        trans = Transformer()
        loss_fn = self.__to_cuda__(CrossEntropyLoss())
        optim = torch.optim.SGD(self.local_module.parameters(), lr=self.learning_rate)
        train_batchs = DataLoader(self.dataset, mini_batch, True)
        
        #epoch轮迭代
        self.local_module.eval()
        print("参与方{}开始训练..".format(self.client_id))
        loss_mean = 0
        total_loss = 0
        #一次迭代
        i = 0
        for image, label in train_batchs:
            #计算输出
            image = self.__to_cuda__(image)
            label = self.__to_cuda__(label)
            output = self.local_module(image)
            #计算损失
            loss = loss_fn(output, label)
            #初始化梯度参数
            optim.zero_grad()
            #反向传播
            loss.backward()
            #self.set_grad(self.local_module.parameters())
            optim.step()
            total_loss += loss
            #end-if
            i += 1
            if(i >= epoch): break
        loss_mean = total_loss / epoch
        print("第{}个参与方{}轮迭代, 损失值：{}".format(self.client_id, i, loss_mean))
        #返回encrypted parameters
        parameters_list = trans.para_to_list(self.local_module.state_dict(), self.local_module)
        return self.sealer.encrypt(parameters_list)

    def test(self, module_path=None):
        mini_batch = 64
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

    def save(self, save_path):
        torch.save(self.local_module.state_dict(), save_path)
        return True