from Module import cifar10
import torch
import torchvision
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from transformer import Transformer
from ctypes import *
import client
from seal import *

if __name__ == "__main__":
    sealer = Seal()
    module = cifar10()
    module1 = cifar10()
    trans = Transformer()

    params_list = trans.para_to_list(module.state_dict(), module)
    params_list1 = trans.para_to_list(module1.state_dict(), module1)
    encrypted_params = sealer.encrypt(params_list)
    encrypted_params1 = sealer.encrypt(params_list1)


    mut_encrypted = sealer.mutiple(encrypted_params, encrypted_params1)
    decrypted_params = sealer.decrypt(mut_encrypted, params_list.__len__())
    print(decrypted_params)