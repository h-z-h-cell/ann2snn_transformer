import spikingjelly.clock_driven.neuron
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.autograd import Function
import random
import os
import numpy as np
import Myneuron
def seed_all(seed=42,rank=0):#设置随机数种子使得实验可重复
    random.seed(seed+rank)
    os.environ['PYTHONHASHSEED'] = str(seed+rank) #禁止hash随机化，使得实验可复现
    np.random.seed(seed+rank)
    torch.manual_seed(seed+rank)
    torch.cuda.manual_seed(seed+rank)
    torch.cuda.manual_seed_all(seed+rank)
    torch.backends.cudnn.deterministic = True #使用确定性卷积算法

class GradFloor(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
myfloor = GradFloor.apply
class MyFloor(nn.Module):#utils中replace_activation_by_floor调用
    def __init__(self, up=8., t=32):#up就是QCFS公式中的阈值\theta,t就是公式中的离散的取值数T
        super().__init__()
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)#第一个参数是初始值
        self.t = t
    def forward(self, x):
        x = x / self.up
        x = myfloor(x*self.t+0.5)/self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        return x

def isActivation(name):#根据名字判断某一个模块是不是激活函数模块
    if 'relu' in name.lower() or 'floor' in name.lower():
        return True
    return False

def replace_activation_by_floor(model,t=8,up=8.):#替换模块中的激活函数为QFCS中的离散函数，参数为t，t=0改用tcl，用于ANN的训练和测试
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_floor(module, t)
        if isActivation(module.__class__.__name__.lower()):
            model._modules[name] = MyFloor(up, t)
    return model

def replace_activation_by_neuron(model,T):#将模型中ann的激活函数为snn神经元的计算方式，用于SNN输出测试
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_activation_by_neuron(module,T)
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "up"):
                model._modules[name] = Myneuron.ScaledNeuron(scale=module.up.item(),T=T)#ScaledNeron就是SNN神经元
            else:#用的不是有离散有上界的relu
                print("wrong")
                model._modules[name] = Myneuron.ScaledNeuron(scale=1.,T=T)
    return model

def replace_maxpool2d_by_avgpool2d(model):#将模型中的平均池化层全部替换为最大池化层
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):#将子模型中递归替换
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model

def replace_maxpool2d_by_MaxpoolNeuron (model):#将模型中的平均池化层全部替换为最大池化层
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):#将子模型中递归替换
            model._modules[name] = replace_maxpool2d_by_MaxpoolNeuron(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = Myneuron.MaxpoolNeuron(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model

def reset_net(model):#将模型中的所以神经元初始化，用于SNN重新初始化
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            reset_net(module)
        if 'Neuron' in module.__class__.__name__:
            module.reset()
    return model

class AverageMeter:#用来计算平均值的类
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):#用来计算top-k准确率的的函数
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
