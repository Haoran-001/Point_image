import torch
import torch.optim as optim
import torch.nn as nn
import os, shutil, json
import argparse
import time
import shutil

from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from tools.utils import record_times
from models.MVCNN import SVCNN, MVCNN

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=3) #  8it will be *12 images in each batch for mvcnn  二阶段batchsize
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000) # 每一类模型数量///双Titan可设置为14， 双RTX2080可设置为8
parser.add_argument("-lr", type=float, help="learning rate", default=1e-4)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0001)  #降低权重
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')    #预训练
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="SCNet101")   #选择cnn模型  默认resnet50
parser.add_argument("-num_views", type=int, help="number of views", default=20)           #视图数量
parser.add_argument("-train_path", type=str, default=r'C:\Users\xdtech\Documents\modelnet40v2png_ori4\modelnet40v2png_ori4/*/train')  #训练数据
parser.add_argument("-val_path", type=str, default=r"C:\Users\xdtech\Documents\modelnet40v2png_ori4\modelnet40v2png_ori4/*/test")     #验证数据
parser.set_defaults(train=False)


def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)     #删除文件
        os.mkdir(log_dir)
    return log_dir


if __name__ == '__main__':
    args = parser.parse_args()
    pretraining = not args.no_pretraining
    # 将MVCNN参数配置保存在config.json中
    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

    print('###################Stage 1####################')
    # 创建MVCNN_stage_1文件夹
    log_dir = args.name+'_stage_1'
    create_folder(log_dir)

    # 导入预训练模型，从预训练模型的参数开始优化训练，采用模型为SCNet
    cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)
    cnet_ = torch.nn.DataParallel(cnet, device_ids=[0])  # 0.1
    n_models_train = args.num_models * args.num_views
    optimizer = optim.Adam(cnet_.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # batch_size:这里原文为64时，1080TiGPU显存不足，我改为32占用比为8.6/11GB
    # 换为TitanXP时，用64可以运行，占用比为 10/12GB, 两个TitanX是为128
    print('Loading traning set and val set!')
    # 没有对3D模型转换下的2维图像进行图像增强，如缩放尺寸或者旋转
    train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=10)
    val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=48, shuffle=False, num_workers=10)

    # 训练集和测试集分别为9843和2468再乖以12
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))

    # 这里只是定义一个训练器，记录数据，输出loss和acc, svcnn和num_view=1即只要单个图像输入
    trainer = ModelNetTrainer(cnet_, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=1)
    tic1 = time.clock()
    trainer.train(n_epochs=0) # 测试时设为1，看能否完整跑完两个阶段
    toc1 = time.clock()
    print('The training time of first stage: %d m' % ((toc1-tic1)/60))

    # STAGE 2
    print('###################Stage 2####################')
    log_dir = args.name+'_stage_2'
    create_folder(log_dir)

    # cnet_2与cnet采用相同的网络
    cnet_2 = MVCNN(args.name, cnet, pool_mode='no', nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views)
    del cnet
    cnet_2 = torch.nn.DataParallel(cnet_2, device_ids=[0]) # 0,1
    optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    # bitch_size: 原文为8,内存不足改为6

    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=10) # shuffle needs to be false! it's done within the trainer
    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=10)

    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))

    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=args.num_views)
    tic2 = time.clock()
    trainer.train(n_epochs=10)  # 测试时设为1，看能否完整跑完两个阶段
    toc2 = time.clock()
    print('The training time of second stage:%d m' % ((toc2-tic2)/60))
    record_times((toc1-tic1)/60, (toc2-tic2)/60, 'records.txt')