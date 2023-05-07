import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        #定义各层参数，详细见forward
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_relu = nn.ReLU(inplace=True)

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_relu = nn.ReLU(inplace=True)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        B,N,C = x.shape
        # torch.Size([64, 64, 384])
        x = self.fc1_linear(x)
        # 1536是隐藏维度为 4*C
        # torch.Size([64, 64, 1536])
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(B, N, self.c_hidden).contiguous()
        # torch.Size([64, 64, 1536])
        x = self.fc1_relu(x)

        x = self.fc2_linear(x)
        # torch.Size([64, 64, 384])
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(B, N, C).contiguous()
        # torch.Size([64, 64, 384])
        x = self.fc2_relu(x)
        return x


class VSA(nn.Module):
    def __init__(self, dim, num_heads=12):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        #定义各层参数，详细见forward
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_relu = nn.ReLU(inplace=True)

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_relu = nn.ReLU(inplace=True)

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_relu = nn.ReLU(inplace=True)
        self.attn_softmax = nn.Softmax(dim=2)

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #这个C就是特征维度D
        B,N,C = x.shape

        #以Cifar10的B=64,T=4,D=384,num_heads=12运算为例
        x_for_qkv = x
        # torch.Size([64, 64, 384])

        #计算Query矩阵
        q_linear_out = self.q_linear(x_for_qkv)
        # torch.Size([64, 64, 384])
        q_linear_out = self.q_bn(q_linear_out. transpose(-1, -2)).transpose(-1, -2).contiguous()
        # torch.Size([64, 64, 384]), permute函数的作用是对tensor进行转置，这里多一个维度是采用多头自注意力机制
        q = q_linear_out.reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3).contiguous()
        # torch.Size([64, 12, 64, 32])
        q = self.q_relu(q)

        # 计算Key矩阵
        k_linear_out = self.k_linear(x_for_qkv)
        # torch.Size([64, 64, 384])
        k_linear_out = self.k_bn(k_linear_out.transpose(-1, -2)).transpose(-1, -2).contiguous()
        # torch.Size([64, 64, 384]), permute函数的作用是对tensor进行转置，这里多一个维度是采用多头自注意力机制
        k= k_linear_out.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        # torch.Size([64, 12, 64, 32])
        k = self.k_relu(k)

        # 计算Value矩阵
        v_linear_out = self.v_linear(x_for_qkv)
        # torch.Size([64, 64, 384])
        v_linear_out = self.v_bn(v_linear_out.transpose(-1, -2)).transpose(-1, -2).contiguous()
        # torch.Size([64, 64, 384]), permute函数的作用是对tensor进行转置，这里多一个维度是采用多头自注意力机制
        v = v_linear_out.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        # torch.Size([64, 12, 64, 32])
        v  = self.v_relu(v)

        #attn为Q*K^T
        #q为torch.Size([64, 12, 64, 32])，k.transpose(-2, -1)为torch.Size([4, 64, 12, 64, 64])
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # torch.Size([64, 12, 64, 64])
        attn  = self.attn_softmax(attn)
        x = attn @ v
        # torch.Size([64, 12, 64, 32])
        # 多头合并
        x = x.transpose(1,2).reshape(B, N, C).contiguous()
        # torch.Size([64, 64, 384])
        x = self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2)
        # torch.Size([64, 64, 384])
        x = self.proj_relu(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        #对单个batch归一化,此处没用上
        self.norm1 = norm_layer(dim)
        self.attn = VSA(dim, num_heads=num_heads)
        #imagenet中在这里还有一个DropPath层，是为了防止过拟合
        #drop_path 将深度学习模型中的多分支结构随机 “失效”，而 drop_out 是对神经元随机 “失效”。
        # 换句话说，drop_out 是随机的点对点路径的关闭，drop_path 是随机的点对层之间的关闭
        self.norm2 = norm_layer(dim)
        #计算mlp隐藏维度,按照论文中为4*D
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        #单个Encoder块，经过SSA自注意力和MLP多功能感知两个过程
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class Embedding(nn.Module):
    def __init__(self, img_size_h=32, img_size_w=32, patch_size=4, in_channels=3, embed_dims=384):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        #后续用于将图像分割为patch_size*patch_size的小块
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        # //是向下取整的运算
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        # #四个Embedding块，4块的卷积层的输出通道数分别是D/8、D/4、D/2、D，cifar10测试时我们采用的embed_dims为384
        # self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        # self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        # self.proj_relu = nn.ReLU(inplace=True)
        #
        # self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        # self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        # self.proj_relu1 = nn.ReLU(inplace=True)
        #
        # self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        # self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        # self.proj_relu2 = nn.ReLU(inplace=True)
        # self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        #
        # self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        # self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        # self.proj_relu3 = nn.ReLU(inplace=True)
        # self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        #相对位置嵌入
        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(in_channels, embed_dims,kernel_size=self.patch_size,stride=self.patch_size,padding='valid')
        self.proj_bn0 = nn.BatchNorm2d(embed_dims)
        self.proj_relu0 = nn.ReLU(inplace=True)
    def forward(self, x):
        # torch.Size([64, 3, 32, 32])
        # 这里的H,W是一个图片的宽高
        B, C, H, W = x.shape

        x = self.conv0(x)
        x = self.proj_bn0(x).contiguous()
        x = self.proj_relu0(x)

        # #以cifar为例记录维度变化过程
        # # torch.Size([64, 3, 32, 32])
        # x = self.proj_conv(x)
        # # torch.Size([64, 48, 32, 32])
        # x = self.proj_bn(x).contiguous()
        # x = self.proj_relu(x)
        #
        # # torch.Size([64, 48, 32, 32])
        # x = self.proj_conv1(x)
        # # torch.Size([64, 96, 32, 32])
        # x = self.proj_bn1(x).contiguous()
        # x = self.proj_relu1(x)
        #
        # # torch.Size([64, 96, 32, 32])
        # x = self.proj_conv2(x)
        # # torch.Size([64, 192, 32, 32])
        # x = self.proj_bn2(x).contiguous()
        # # torch.Size([64, 192, 32, 32])
        # x = self.proj_relu2(x)
        # x = self.maxpool2(x)
        # # torch.Size([64, 192, 16, 16])
        #
        # # torch.Size([64, 192, 16, 16])
        # x = self.proj_conv3(x)
        # # torch.Size([64, 384, 16, 16])
        # x = self.proj_bn3(x).contiguous()
        # x = self.proj_relu3(x)
        # # torch.Size([64, 384, 16, 16])
        # x = self.maxpool3(x)
        # # torch.Size([64, 384, 8, 8])

        # torch.Size([64, 384, 8, 8])
        x_feat = x.contiguous()
        # torch.Size([64, 384, 8, 8])
        x = self.rpe_conv(x)
        # torch.Size([64, 384, 8, 8])
        x = self.rpe_bn(x).contiguous()
        # torch.Size([64, 384, 8, 8])
        x = self.rpe_relu(x)
        x = x + x_feat
        # torch.Size([64, 384, 8, 8])
        # B,N,C(这里的C就是特征维度D)
        x = x.flatten(-2).transpose(-1, -2)
        # torch.Size([64, 64, 384])
        return x


class Transformer(nn.Module):
    def __init__(self,
                 img_size_h=32, img_size_w=32, patch_size=16, in_channels=3, num_classes=10,
                 embed_dims=384, num_heads=12, mlp_ratios=4, depths=4, T = 4
                 ):
        #img_size是图像的大小，in_channels是通道数，num_classes是分类数
        #embed_dims是维度D，num_heads是多头自注意力的头数，mlp_ratios是mlp隐藏维度的倍数，depths是Block块的个数，T是时间步长
        #drop_path_rate=0
        #patch_size用于将图像分割为patch_size*patch_size的小块因为是cifar10的小数据集暂时还没加上这个功能
        # 调用nn.Module的初始化
        super().__init__()
        self.T = T  # time step
        self.num_classes = num_classes
        self.depths = depths
        self.patch_size=patch_size
        # patch_embed为一个Spiking Patch Spliting模块
        patch_embed = Embedding(img_size_h=img_size_h,
                                img_size_w=img_size_w,
                                patch_size=patch_size,
                                in_channels=in_channels,
                                embed_dims=embed_dims)
        # block为L个Spikformer Encoder Block模块的串联
        block = nn.ModuleList([Block(dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios)
            for j in range(depths)])

        # 为模块赋值patch_embed和block属性
        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        # nn.Identity()是恒等变换函数
        # 定义最后分类的全连接层
        # embed_dims是特征的维度
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        # 模型赋值初始参数
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # timm.models.layers.trunc_normal_
            # 将参数初始化为截断正态分布
            nn.init.trunc_normal_(m.weight, std=.02)
            # 初始化全连接层偏置为0
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # 初始化偏置为1
            nn.init.constant_(m.bias, 0)
            # 初始化连接权重为1
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        # 运行SPS层
        # torch.Size([64, 3, 32, 32])
        x = patch_embed(x)
        # torch.Size([64, 64, 384])
        # 运行L个Spikformer Encoder Block层
        for blk in block:
            x = blk(x)
        # torch.Size([64, 64, 384])
        # B*N*D
        # torch.Size([64, 64, 384])
        # 按照第三个维度取平均值
        return x.mean(1)

    def forward(self, x):
        # 一个样本的维度[B, C, W, H]
        # torch.Size([64, 3, 32, 32])
        # 经过SPS、多个SSA和MLP提取特征
        x = self.forward_features(x)
        # [B,D]
        # torch.Size([64, 384])
        x = self.head(x)
        # torch.Size([64, 10])
        return x



