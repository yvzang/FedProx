from client import client
from CNN import CNN
from torch.utils.tensorboard import SummaryWriter
import torch
from threading import Thread


writer = SummaryWriter(log_dir="./runs/result")
def train(clients_num):
    clients = [client(i) for i in range(clients_num)]
    parameters = clients[0].get_parameters()
    i = 1
    while(True):
        torch.cuda.empty_cache()
        print("第{}轮训练开始..".format(i))
        client_kl = []
        #计算kl散度
        for c in clients:
            client_kl.append((c, c.calculate_kl_div()))
        #排序
        client_kl.sort(key=lambda x:x[1])
        #剪枝
        kl_head = client_kl[0][1] + client_kl[0][1] * 0.3
        for c in client_kl:
            if(c[1] > kl_head ):client_kl.remove(c)
        client_sorted = [c[0] for c in client_kl]
        #连接
        for c_n in range(client_kl.__len__() - 1):
            client_sorted[c_n].set_next(client_sorted[c_n + 1])
        #串行训练
        for cl in clients:
            cl.set_parameters(parameters)
            loss = cl.update_parameters()
            writer.add_scalars(main_tag="train/loss", 
                               tag_scalar_dict={
                                   "train": loss
                               },
                               global_step=i)
            parameters = cl.get_parameters()
            
        #test
        if(i % 10 == 0):
            cl = clients[0]
            cl.set_parameters(parameters)
            accu = cl.test()
            writer.add_scalars(main_tag="train/accu", 
                                tag_scalar_dict={
                                    "train": accu
                                },
                                global_step=i)
            if(accu > 0.8):
                writer.close()
                break
        i = i + 1

def train_without_kldiv(clients_num):
    clients = [client(i) for i in range(clients_num)]
    parameiters = clients[0].get_parameters()
    i = 1
    while(True):
        print("第{}轮训练开始..".format(i))
        for cl in clients:
            cl.calculate_kl_div()
            loss = cl.set_parameters(parameiters)
            writer.add_scalars(tag="train/loss", 
                    tag_scalar_dict={
                        "train_without_kldiv": loss
                    },
                    global_step=i)
            cl.update_parameters()
            parameiters = cl.get_parameters()
        #test
        if(i % 10 == 0):
            cl = clients[0]
            accu = cl.set_parameters(parameiters)
            writer.add_scalars(tag="train/accu", 
                    tag_scalar_dict={
                        "train_without_kldiv": accu
                    },
                    global_step=i)
            if(accu >= 0.8):
                writer.close()
                break
        i = i + 1


if __name__ == "__main__":
    train(4)