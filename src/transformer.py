import torch
import numpy as np

class Transformer():
    def __init__(self) -> None:
        pass

    def para_to_list(self, parameters, module) -> list:
        '''参数构造成一个列表'''
        result_lst = []
        if isinstance(module, torch.nn.Module) == False or isinstance(parameters, dict) == False:
            raise ValueError("模型参数类型不正确")
        for key, value in parameters.items():
            value = value.cpu()
            value = value.reshape([-1])
            lst = value.numpy().tolist()
            result_lst = result_lst + lst
        return result_lst
    
    def list_to_para(self, lst, module):
        '''将一个列表lst构造成模型module的参数dict类型'''
        if isinstance(module, torch.nn.Module) == False:
            raise ValueError("模型参数类型不正确")
        para_dict = module.state_dict()
        for key, value in para_dict.items():
            value = value.cpu()
            field = np.prod(list(value.shape))
            field_lst = lst[:field]
            lst = lst[field:]
            tens = torch.Tensor(field_lst).reshape(value.shape)
            para_dict[key] = tens
        return para_dict

