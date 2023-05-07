import torch
from spikingjelly.clock_driven.neuron import *
from spikingjelly.clock_driven.surrogate import *
def heaviside(x: torch.Tensor,y: torch.Tensor,z: int):
    return ((x>=0)&(y<z)).to(x)
def heaviside2(x: torch.Tensor,y: torch.Tensor):
    return  (((y>=1)&(x>=0))).to(x)
class TwotypeSpikeIFNode(IFNode):
    def __init__(self,T, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, cupy_fp32_inference=False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        if cupy_fp32_inference:
            check_backend('cupy')
        self.cupy_fp32_inference = cupy_fp32_inference
        self.all_fire=None
        self.Tmax = T
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x
    def neuronal_fire(self):
        if self.all_fire==None:
            self.all_fire=torch.zeros_like(self.v)
        #all_fire记录总发放率，要维持在0-Tmax之间
        tmp = heaviside(self.v-self.v_threshold,self.all_fire,self.Tmax)-heaviside2(-self.v_threshold-self.v,self.all_fire)
        self.all_fire=self.all_fire+tmp
        return tmp
class ScaledNeuron(nn.Module):#utils中replace_activation_by_neurond调用，作为SNN的神经元
    def __init__(self,T,scale=1.):
        super(ScaledNeuron, self).__init__()
        self.scale = scale
        self.t = 0 #当前模拟的时间步长t
        self.neuron = TwotypeSpikeIFNode(T,v_reset=None)#调用可以发放负脉冲的TwotypeSpikeIFNode,T是允许的最大脉冲数
        # self.neuron = spikingjelly.clock_driven.neuron.IFNode(v_reset=None)
        self.neuron.all_fire=None
    def forward(self, x):
        x = x / self.scale
        if self.t == 0:
            self.neuron(torch.ones_like(x)*0.5)
        x = self.neuron(x)
        self.t += 1
        return x * self.scale
    def reset(self):#重置神经元初始电位
        self.t = 0
        self.neuron.all_fire=None
        self.neuron.reset()
class MaxpoolNeuron(nn.MaxPool2d):#utils中replace_maxpool2d_by_MaxpoolNeuron调用，作为SNN最大池化层的替代神经元
    def __init__(self,**keywords):
        super(MaxpoolNeuron, self).__init__(**keywords)
        self.tot = 0
        self.last = 0
        self.now = 0
    def forward(self, x):
        self.tot += x
        self.last = self.now
        self.now= F.max_pool2d(self.tot, self.kernel_size, self.stride,
                     self.padding, self.dilation, ceil_mode=self.ceil_mode,
                     return_indices=self.return_indices)
        return self.now-self.last
    def reset(self):#重置神经元初始电位
        self.tot = 0
        self.last = 0
        self.now = 0

