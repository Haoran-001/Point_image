import numpy as np
import torch
import torch.nn as nn
import os, shutil
import argparse
import scipy.io as sio
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=6) #  8it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="resnet50x")
parser.add_argument("-num_views", type=int, help="number of views", default=6)
parser.add_argument("-sketch_path", type=str, default='F:\CesareDou\Sketch2ModelNet')
# E:/CesareDou/mvcnn_pytorch-master/modelnet40_images_new_12x
# E:/CesareDou/ModelNet40_images_2048_224_6views
parser.set_defaults(train=False)


def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)


def extract_features(model, data_loader):
    all_correct_points = 0
    all_points = 0
    wrong_class = np.zeros(40)
    samples_class = np.zeros(40)

    model.eval()
    all_target = []
    all_pred = []

    features_list = []
    labels_list = []
    # print(model)

    data_loader = tqdm(data_loader, desc='Extract Features')

    for _, data in tqdm(enumerate(data_loader)):
        N, V, C, H, W = data[1].size()
        in_data = Variable(data[1]).view(-1, C, H, W).cuda()
        target = Variable(data[0]).cuda()
        features, out_data = model(in_data)

        for i in range(target.size()[0]):
            labels_list.append(target[i].cpu().detach().numpy())
            features_list.append(features[i].cpu().detach().numpy())

        pred = torch.max(out_data, 1)[1]
        results = pred == target

        for i in range(results.size()[0]):
            if not bool(results[i].cpu().data.numpy()):
                wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
            samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
        correct_points = torch.sum(results.long())

        target = target.cpu().data.numpy()
        pred = pred.cpu().data.numpy()
        for j in range(len(pred)):
            all_target.append(target[j])
            all_pred.append(pred[j])

        all_correct_points += correct_points
        all_points += results.size()[0]

    print('Total # of 3D models: ', all_points)
    val_mean_class_acc = np.mean((samples_class - wrong_class) / samples_class)
    acc = all_correct_points.float() / all_points
    val_overall_acc = acc.cpu().data.numpy()
    cm = confusion_matrix(all_pred, all_target)

    print('OA: ', val_overall_acc)
    print('AA: ', val_mean_class_acc)

    return features_list, labels_list


args = parser.parse_args()
print('###################Retrieval####################')
# 创建MVCNN_stage_1文件夹
log_dir = args.name + '_retrieval'
create_folder(log_dir)

sketch_dataset = MultiviewImgDataset(args.sketch_path, scale_aug=False, rot_aug=False, test_mode=True,
                                      shuffle=False, num_views=args.num_views)
sketch_loader = torch.utils.data.DataLoader(sketch_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
print('num_gallery_files:  ' + str(len(sketch_dataset.filepaths)))

cnet = SVCNN(args.name, nclasses=40, pretraining=False, cnn_name=args.cnn_name)
cnet_2 = MVCNN(args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views)
del cnet

model_parameters = torch.load('records/model_4.pth')
cnet_2 = torch.nn.DataParallel(cnet_2)
torch.backends.cudnn.benchmark = True
cnet_2.load_state_dict(model_parameters)

features_list, gallery_labels_list = extract_features(cnet_2, gallery_loader)
sio.savemat(log_dir + '/sketch_features.mat', {'features': features_list})

