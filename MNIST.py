#!/usr/bin/python3
# coding: utf-8

import os
import glob
import time
import random
import argparse
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image 

def load_args(default_config=None):
    parser = argparse.ArgumentParser(description = 'Pytorch-MNIST_classification')
    # directory
    parser.add_argument('--data_dir',type=str, default='./data/', help='download and load data directory')
    parser.add_argument('--save_visual_source_dir',type=str, default='./result/visual_source/', help='train visual source data directory')
    parser.add_argument('--save_chart_dir',type=str, default='./result/chart/', help='save chart directory')
    parser.add_argument('--save_heatmap_dir',type=str, default='./result/heatmap/', help='save heatmap directory')
    parser.add_argument('--save_model_dir',type=str, default='./result/model/', help='save model directory')
    parser.add_argument('--save_log_dir',type=str, default='./result/log/', help='train log directory')
    # model
    parser.add_argument('--model_name',type=str, default='Model', choices=['Model','Model1','Model2'], help='model name')
    # config
    parser.add_argument('--device_ids',type=list, default=[6,7], help='ids of device')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='size of each image batch' )
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum')
    parser.add_argument('--criterion', type=str, default='CE', choices=['CE','BCE'], help='Loss')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam','SGD','AdamW'], help='optimizer')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
    # train or test
    parser.add_argument("--train", default=True, action='store_false', help="train or test")
    parser.add_argument("--test_epoch", type=int, default=-1, help="which epoch of model to test")
    parser.add_argument('--pretrained_weights', type=str, default='checkpoints/', help='if specified starts from checkpoint model')
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    # other config
    parser.add_argument("--workers", type=int, default=2, help="dataloder thread numbers")
    args = parser.parse_args()
    return args

args=load_args()
print(args)

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.benchmark = True

def new_folder():
    # 建立保存可视化原图片的文件夹
    visual_source_folder=args.save_visual_source_dir
    if not os.path.exists(visual_source_folder):
        os.makedirs(visual_source_folder)
    # 建立存放折线图的文件夹
    chart_folder=os.path.join(args.save_chart_dir, args.model_name)
    if not os.path.exists(chart_folder):
        os.makedirs(chart_folder)
    # 建立存放训练模型的文件夹
    model_folder=os.path.join(args.save_model_dir, args.model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    # 建立存放训练日志的文件夹和文件
    if not os.path.exists(args.save_log_dir):
        os.makedirs(args.save_log_dir)
    log_name=os.path.join(args.save_log_dir, '{}_train_log.txt'.format(args.model_name))
    if not os.path.exists(log_name):
        os.mknod(log_name)

def data_processing():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])#mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]
    data_train = datasets.MNIST(root = args.data_dir,transform=transform,train = True,download = True)
    data_test = datasets.MNIST(root= args.data_dir,transform = transform,train = False)

    # 这里注意batch size要对应放大倍数
    train_loader = torch.utils.data.DataLoader(dataset=data_train,batch_size = args.batch_size * len(args.device_ids),shuffle = True,num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(dataset=data_test,batch_size = args.batch_size * len(args.device_ids),shuffle = True,num_workers=args.workers)

    return train_loader, test_loader

def visual_source(test_loader):
    # 3.2.3　可视化源数据
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    # 保存单张
    fig = plt.figure()
    plt.imshow(example_data[0][0], cmap='gray', interpolation='none')
    # save_visual_source_path = './result/visual_source/)
    plt.savefig(os.path.join(args.save_visual_source_dir,'single_img_{}.jpg'.format(example_targets[0])))

    # 保存多张
    fig = plt.figure() #也可设置画布大小：figsize=(10, 10)
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig(os.path.join(args.save_visual_source_dir,'multi_img.jpg'))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 =  nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(stride=2, kernel_size=2),
                                    )
        self.dense =  nn.Sequential(nn.Linear(14 * 14 * 128, 1024),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(1024, 10)
                                    )
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(196 * 8 * 8, 1024)#256 * 8 * 8
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def define_model():
    if args.model_name == 'Model':
        model = Model()
    elif args.model_name == 'Model1':
        model = Model1()
    # elif args.model_name == 'Model2':
    #     model = Model2()
    model = torch.nn.DataParallel(model, device_ids=args.device_ids) # 声明所有可用设备
    model = model.cuda(device=args.device_ids[0]) # 模型放在主设备
    return model

# define criterion and optimizer of model
def define_criterion_optimizer(model):
    # define criterion
    if args.criterion=='CE': criterion = torch.nn.CrossEntropyLoss()
    # define optimizer
    if args.optimizer=='Adam':
        optimizer = torch.optim.Adam(model.parameters())#,lr=0.01
    elif args.optimizer=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    return criterion, optimizer

def visualization(epoch,train_losses,train_acces,eval_losses,eval_acces):
    fig = plt.figure()
    plt.title('Epoch {} Train/Test loss and acc'.format(epoch))
    plt.plot(np.arange(len(train_losses)), train_losses, color='#ffcea9', label='train_losses') #color=palette(3), marker='*'
    plt.plot(np.arange(len(train_acces)), train_acces, color='#aad4ff', label='train_acces')#, marker='^'
    plt.plot(np.arange(len(eval_losses)), eval_losses, color='#fe9d52', label='eval_losses')#, marker='s'
    plt.plot(np.arange(len(eval_acces)), eval_acces, color='#46a4ff', label='eval_acces')#, marker='o'
    plt.legend()  # 让图例生效 # plt.legend(['Train Loss'], loc='upper right')
    save_chart_path = os.path.join(args.save_chart_dir, args.model_name, '{}_chart_epoch_{}.jpg'.format(args.model_name,epoch))
    plt.savefig(save_chart_path)
    plt.show()

# def show_heatmap(epoch,test_loader):
#     device = torch.device('cpu')
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     net = Model()
#     net = torch.nn.DataParallel(net)
#     net = net.to(device)
#     # net = net.cuda(device=args.device_ids[0])
#     model_path=args.save_model_dir+'/model_parameter_epoch_{}.pth'.format(epoch)
#     net.load_state_dict(torch.load(model_path))
    
#     # net.load_state_dict(torch.load(model_path, map_location=device))  # 载入训练的resnet模型权重，你将训练的模型权重放到当前文件夹下即可
#     # net = torch.nn.DataParallel(net, device_ids=args.device_ids)
#     # net = net.to(device)
    
#     if isinstance(net,torch.nn.DataParallel):
#         net = net.module

#     target_layers = [net.dense[-1]] #这里是 看你是想看那一层的输出，我这里是打印的resnet最后一层的输出，你也可以根据需要修改成自己的
#     print(target_layers)
#     # data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#     # data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])#mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]
#     # 导入图片
#     # img_path = "./38.jpg"
#     # image_size = 500#训练图像的尺寸，在你训练图像的时候图像尺寸是多少这里就填多少
#     # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
#     # img = Image.open(img_path).convert('RGB')#将图片转成RGB格式的
#     # img = np.array(img, dtype=np.uint8) #转成np格式
#     # img = center_crop_img(img, image_size) #将测试图像裁剪成跟训练图片尺寸相同大小的

#     # [C, H, W]
#     # img = Image.open(img_path)

#     examples = enumerate(test_loader)
#     batch_idx, (example_data, example_targets) = next(examples)
#     img = example_data
#     img=img.to(device)
#     # img.cuda(device=args.device_ids[0])

#     # img_tensor = data_transform(img)#简单预处理将图片转化为张量
    
#     # expand batch dimension
#     # [C, H, W] -> [N, C, H, W]
#     # input_tensor = torch.unsqueeze(img_tensor, dim=0) #增加一个batch维度
#     input_tensor=img
#     input_tensor=input_tensor.to(device)
#     # input_tensor=input_tensor.unsqueeze(0)#.unsqueeze(0)# [1,1,28,28]
#     print(input_tensor.shape)
#     # input_tensor.cuda(device=args.device_ids[0])
#     cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)
#     # cam.cuda(device=args.device_ids[0])
#     print(input_tensor.shape)
#     grayscale_cam = cam(input_tensor=input_tensor)

#     grayscale_cam = grayscale_cam[0, :]
#     visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
#                                       grayscale_cam,
#                                       use_rgb=True)
#     plt.imshow(visualization)
#     plt.savefig('./result/heatmap/1.png')#将热力图的结果保存到本地当前文件夹
#     plt.show()

def save_model(epoch,model):
    save_model_path = os.path.join(args.save_model_dir, args.model_name, '{}_parameter_epoch_{}.pth'.format(args.model_name,epoch))
    torch.save(model.state_dict(), save_model_path)

def load_model(model_path):
    checkpoint = torch.load(model_path)
    test_model=define_model()
    test_model.load_state_dict(checkpoint)#checkpoint['model']
    return test_model

def save_logger(epoch_result):
    log_name=os.path.join(args.save_log_dir, '{}_train_log.txt'.format(args.model_name))
    to_write = "Time: {time_str} {epoch_result}\n".format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                                          epoch_result=epoch_result)
    with open(log_name, "a") as f:
        f.write(to_write)

def load_log_loss_acc():
    # 加载log中每个epoch的loss和acc，返回用于存放的变量
    train_losses=[]
    train_acces=[]
    eval_losses=[]
    eval_acces=[]

    log_name=os.path.join(args.save_log_dir, '{}_train_log.txt'.format(args.model_name))
    file = open(log_name, 'r', encoding='UTF-8')
    for line in file.readlines():
        s = line.strip().split(': ')[-4:]
        for i, s1 in enumerate(s):
            s2 = float(s1.split(',')[0])
            if i==0:
                train_losses.append(s2)
            elif i==1:
                train_acces.append(s2)
            elif i==2:
                eval_losses.append(s2)
            else:
                eval_acces.append(s2)
    file.close()
    return train_losses,train_acces,eval_losses,eval_acces

def train(model, strat_epoch, train_loader, test_loader):
    criterion, optimizer=define_criterion_optimizer(model)
    
    train_losses,train_acces,eval_losses,eval_acces=load_log_loss_acc()
    # # 存放每个epoch的loss和acc
    # train_losses=[]
    # train_acces=[]
    # eval_losses=[]
    # eval_acces=[]

    for epoch in range(strat_epoch, args.epochs):
        print("-"*10)
        print("Epoch {}/{}".format(epoch, args.epochs))

        #动态修改参数学习率
        if args.optimizer=='SGD':
            if epoch%5==0:
                optimizer.param_groups[0]['lr']*=0.1

        # 在训练集上训练效果
        train_loss = 0
        train_acc = 0
        # 将模型改为训练模式
        model.train()
        for img, label in tqdm(train_loader):
            img, label = img.cuda(device=args.device_ids[0]), label.cuda(device=args.device_ids[0])# 或者是img = img.to(device=args.device_ids[0])
            # 前向传播
            out = model(img)
            loss = criterion(out, label)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录误差
            train_loss += loss.item()
            # 计算分类的准确率
            _, pred = out.max(1) #返回out中，(1)每行(0)每列最大值的(_)值和(pred)索引
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            train_acc += acc
        train_losses.append(train_loss / len(train_loader))
        train_acces.append(train_acc / len(train_loader))
        
        # 在测试集上检验效果
        eval_loss = 0
        eval_acc = 0
        # 将模型改为预测模式
        model.eval()
        for img, label in tqdm(test_loader):
            img, label = img.cuda(device=args.device_ids[0]), label.cuda(device=args.device_ids[0])# 或者是img = img.to(device=args.device_ids[0])
            out = model(img)
            loss = criterion(out, label)
            # 记录误差
            eval_loss += loss.item()
            # 记录准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            eval_acc += acc
        eval_losses.append(eval_loss / len(test_loader))
        eval_acces.append(eval_acc / len(test_loader))
        
        # 周期训练结果
        epoch_result='Model Name: {}, Epoch: {:>2d}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f},'.format(args.model_name, epoch, train_loss / len(train_loader), train_acc / len(train_loader),eval_loss / len(test_loader), eval_acc / len(test_loader))
        # 打印
        print(epoch_result)
        # 记录
        save_logger(epoch_result)
        # 可视化
        visualization(epoch,train_losses,train_acces,eval_losses,eval_acces)
        # 保存epoch模型
        save_model(epoch,model)
        # 用grad-cam生成heatmap
        # img_path='./result/visual_source/single_img0_3.jpg'
        # show_heatmap(epoch,test_loader)

def test(test_epoch,test_loader):
    model_path=os.path.join(args.save_model_dir,args.model_name,'{}_parameter_epoch_{}.pth'.format(args.model_name,test_epoch))
    test_model=load_model(model_path)
    
    criterion, _ = define_criterion_optimizer(test_model)

    test_loss = 0
    test_acc = 0
    test_model.eval()
    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img, label = img.cuda(device=args.device_ids[0]), label.cuda(device=args.device_ids[0])# 或者是img = img.to(device=args.device_ids[0])
            out = test_model(img)
            loss = criterion(out, label)
            # 记录误差
            test_loss += loss.item()
            # 记录准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            test_acc += acc
    # 测试结果
    test_result='Test Epoch: {}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(test_epoch, test_loss / len(test_loader), test_acc / len(test_loader))
    # 打印
    print(test_result)

def main():
    # 新建文件夹
    new_folder()
    # 数据预处理和可视化
    train_loader, test_loader=data_processing()
    visual_source(test_loader)
    # 训练
    if args.train:
        # 确定start_epoch
        # 选模型的最新版本
        model=define_model()
        folder=os.path.join(args.save_model_dir, args.model_name)
        model_num = len(os.listdir(folder))# 或者model_num = len(glob.glob(os.path.join(folder,'*.pth')))
        strat_epoch = model_num
        
        if model_num == 0:
            strat_epoch = 0            
        else:
            start_epoch = model_num-1
            strat_model_path = os.path.join(folder, '{}_parameter_epoch_{}.pth'.format(args.model_name,start_epoch))
            checkpoint = torch.load(strat_model_path)
            model.load_state_dict(checkpoint)
        train(model, strat_epoch, train_loader, test_loader)
    # 测试
    else:
        # 用第几次训练后的模型测试
        if args.test_epoch == -1: # 没有指定epoch,默认用最新的
            folder=os.path.join(args.save_model_dir, args.model_name)
            model_num = len(os.listdir(folder))# 或者model_num = len(glob.glob(os.path.join(folder,'*.pth')))
            test_epoch = model_num-1
        else:
            test_epoch = args.test_epoch
        test(test_epoch,test_loader)

if __name__ == '__main__':
    main()
