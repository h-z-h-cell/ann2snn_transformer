import argparse
import yaml
import random
import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from utils import *
import model
import time
from apex import amp
config_parser = argparse.ArgumentParser(description='parameter file')
config_parser.add_argument('-c', '--config', default='cifar10.yml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(description='Training Config')
parser.add_argument('--gpus', default=1, type=int, help='GPU number to use.')
parser.add_argument('--bs', default=64, type=int, help='Batchsize')
parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epochs', default=120, type=int, help='Training epochs')  # better if set to 300 for CIFAR dataset

parser.add_argument('--id', default=None, type=str, help='Model identifier')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--use-amp', type=bool, default=False)

parser.add_argument('--l', default=8, type=int, help='L')
parser.add_argument('--t', default=16, type=int, help='T')
parser.add_argument('--extra-t', default=0, type=int, help='T')
parser.add_argument('--use-maxpool', type=bool, default=True)
parser.add_argument('--mode', type=str, default='ann')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--save-acc', type=bool, default=True)
parser.add_argument('--inherit', type=bool, default=True)
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='transformer')

parser.add_argument('--dim', type=int, default=None, metavar='N',
                    help='embedding dimsension of feature')
parser.add_argument('--num_heads', type=int, default=None, metavar='N',
                    help='attention head number')
parser.add_argument('--patch-size', type=int, default=None, metavar='N',
                    help='Image patch size')
parser.add_argument('--mlp-ratio', type=int, default=None, metavar='N',
                    help='expand ration of embedding dimension in MLP block')

def _parse_args():
    # parse_known_args方法的作用就是当仅获取到基本设置时，如果运行命令中传入了之后才会获取到的其他配置，不会报错；而是将多出来的部分保存起来，留到后面使用
    args_config, remaining = config_parser.parse_known_args()
    #如果有可用的文件，这里默认是”cifar10.yml“，就读入来用
    if args_config.config:
        with open(args_config.config, 'r') as f:
            #解析基本的yaml标记，得到名为cfg的一个字典
            cfg = yaml.safe_load(f)
            #把从文件得到的数据传入parser解析对象
            parser.set_defaults(**cfg)

    # 如果上面的if里面的执行了就将文件参数替换默认的参数设置，否则采用前面设置的默认参数
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    # 将最新参数缓存为文本字符串，以便稍后将它们保存在输出目录中
    # args.__dict__字典形式的配置参数
    # 加入default_flow_style=False这个参数以后，重新写入后的格式跟源文件的格式就是同样的 yaml 风格了
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text
def train(model,train_loader,eval_loader,args):
    if args.save_acc == True:
        if not os.path.exists('./{}'.format(args.id)):
            os.mkdir('./{}'.format(args.id))
        with open("./{}/{}".format(args.id, 'acc_log.txt'), 'a') as f:
            f.write(args_text.replace("\n"," ")+'\n')
    model.to(device)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters() ,lr=args.lr,momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,eta_min=1e-5)
    if args.inherit==True:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    best_acc = 0
    for epoch in range(args.epochs):
        loss1,batch_time1 = train_one_epoch(model,train_loader,optimizer,criterion,args,epoch)
        print("Epoch{}/{} train_losses:{} train_batch_time:{}".format(epoch+1,args.epochs,loss1,batch_time1))
        loss2,top1,top5,batch_time2 = validate(model,eval_loader,criterion,args)
        print("Epoch{}/{} eval_losses:{} top1-acc:{} top5-acc:{} eval_batch_time:{}".format(epoch+1,args.epochs,loss2,top1,top5,batch_time2))
        lr_scheduler.step()
        if args.save == True:
            if not os.path.exists('./{}'.format(args.id)):
                os.mkdir('./{}'.format(args.id))
            if args.use_amp:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict()
                }
            else:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
            torch.save(checkpoint, './{}/model{}.pt'.format(args.id,epoch))
            if top1 > best_acc:
                best_acc = top1
                print("Update-Best:%d" % (epoch + 1))
                torch.save(checkpoint, './{}/model{}.pt'.format(args.id,epoch))
        if args.save_acc == True:
            with open("./{}/{}".format(args.id,'acc_log.txt'),'a') as f:
                f.write("Epoch{}/{} train_losses:{} train_batch_time:{}\n".format(epoch+1,args.epochs,loss1,batch_time1))
                f.write("Epoch{}/{} eval_losses:{} top1-acc:{} top5-acc:{} eval_batch_time:{}\n".format(epoch+1,args.epochs,loss2,top1,top5,batch_time2))

def eval(model,model_path,eval_loader,args):
    model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    criterion = nn.CrossEntropyLoss().cuda()
    if args.mode=='ann':
        loss,top1,top5,batch_time = validate(model,eval_loader,criterion,args)
        print("ann : eval_losses:{} top1-acc:{} top5-acc:{} eval_batch_time:{}".format(loss,top1,top5,batch_time))
    elif args.mode=='snn':
        if args.use_maxpool:
            model = replace_maxpool2d_by_MaxpoolNeuron(model)
        model = replace_activation_by_neuron(model,args.t)
        print("start")
        loss,top1,top5,batch_time = validate_snn(model,eval_loader,criterion,args)
        print("snn : eval_losses:{} top1-acc:{} top5-acc:{} eval_batch_time:{}".format(loss,top1,top5,batch_time))
def train_one_epoch(model, loader, optimizer, loss_fn, args,nowepoch):
    # timm.utils.AverageMeter：用于统计某个数据在一定次数内的平均值和总个数
    # 统计每批平均用时
    batch_time_m = AverageMeter()
    # 统计平均损失
    losses_m = AverageMeter()
    # 运行 model.train()之后，相当于告诉了 BN 层，对之后输入的每个 batch 独立计算其均值和方差，BN层的参数是在不断变化的
    # model.eval() ，相当于告诉 BN 层，我现在要测试了，你用刚刚统计的 μ和 σ来测试我，不要再变了。
    model.train()
    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        # 如果没有预取，就手动载入gpu
        input, target = input.cuda(), target.cuda()
        output = model(input)
        loss = loss_fn(output, target)
        losses_m.update(loss.item(), input.size(0))
        #清空过往梯度
        optimizer.zero_grad()
        # 反向传播，计算当前梯度
        if args.use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        #根据梯度更新网络参数
        optimizer.step()
        #更新每跑一批所用的平均时间
        batch_time_m.update(time.time() - end)
        # 如果是最后一批，或者想每隔几轮打印一些信息就进入这循环
        end = time.time()
        if (batch_idx+1) % int(len(loader)/5) ==0:
            print("Epoch {}/{} train batch {}/{} loss_avg:{} batch_time_avg:{}".format(nowepoch+1,args.epochs,batch_idx,len(loader),losses_m.avg,batch_time_m.avg))
    return losses_m.avg,batch_time_m.avg
#该部分与train_one_epoch大多相似，只挑其中重要的进行解释
def validate(model, loader, loss_fn, args):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    #top1准确度的平均值
    top1_m = AverageMeter()
    #top5准确度的平均值
    top5_m = AverageMeter()
    # 运行 model.train()之后，相当于告诉了 BN 层，对之后输入的每个 batch 独立计算其均值和方差，BN层的参数是在不断变化的
    # model.eval() ，相当于告诉 BN 层，我现在要测试了，你用刚刚统计的 μ和 σ来测试我，不要再变了。
    model.eval()
    end = time.time()
    #不进行梯度更新地计算，否则会自动进行计算图的生成（计算过程的构建，以便梯度反向传播等操作）
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = loss_fn(output, target)
            #计算top1准确率，和top5准确率
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses_m.update(loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()
    return losses_m.avg,top1_m.avg,top5_m.avg,batch_time_m.avg
def validate_snn(model,loader,loss_fn,args):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    model.eval()
    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            output = None
            for tt in range(args.t):  # 模拟sim_len个时间长度
                out = model(input)  # 计算当前输入在当前网络下的输出
                if output!=None:
                    output += out  # 计算目前总脉冲数
                else: output=out
            tmp = torch.zeros_like(input).cuda()
            for tt in range(args.extra_t):  # 模拟extra_t个额外时间长度释放脉冲
                out = model(tmp)  # 计算当前输入在当前网络下的输出
                output += out  # 计算目前总脉冲数
            reset_net(model)  # 更新模型,包括膜电位在内的参数
            loss = loss_fn(output, target)
            # functional.reset_net(model)
            #计算top1准确率，和top5准确率
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses_m.update(loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()
    return losses_m.avg,top1_m.avg,top5_m.avg,batch_time_m.avg
if __name__ == '__main__':
    args, args_text = _parse_args()
    seed_all()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(args_text)
    if args.data=='cifar10':
        from augment import Cutout, CIFAR10Policy
        trans_train = transforms.Compose([transforms.RandomCrop(32, padding=4),  # 在一个随机位置上对图像进行裁剪，32会被处理成(32,32)的大小
                                          transforms.RandomHorizontalFlip(),  # 随机水平翻转，默认概率0.5
                                          CIFAR10Policy(),  # 随机选择CIFAR10上最好的25个子策略之一。
                                          transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
                                          Cutout(n_holes=1, length=16)
                                          ])
        trans_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        train_dataset = torchvision.datasets.CIFAR10(root=".", train=True ,transform=trans_train, download=True)
        eval_dataset = torchvision.datasets.CIFAR10(root=".", train=False ,transform=trans_test, download=True)
        args.img_size=32
        args.num_classes=10
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.bs, shuffle=True)
    model = model.Transformer(img_size_h=args.img_size, img_size_w=args.img_size,
        patch_size=args.patch_size, embed_dims=args.dim, num_heads=args.num_heads, mlp_ratios=args.mlp_ratio,
        in_channels=3, num_classes=args.num_classes)
    if not args.use_maxpool:
        model = replace_maxpool2d_by_avgpool2d(model)#使用平均池化
    model = replace_activation_by_floor(model,t=args.l)
    print("Create model successfully")
    # Loss and optimizer
    if args.action=='train':
        train(model,train_loader,eval_loader,args)
    elif args.action=='test':
        eval(model,args.model_path,eval_loader,args)
#snn : eval_losses:10.04379312210083 top1-acc:72.74 top5-acc:98.05 eval_batch_time:0.8732917627711205

