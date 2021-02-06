import h5py
import numpy as np
from PIL import Image
import json
import os
import math


def sin(x):
    return math.sin(math.radians(x))


def cos(x):
    return math.cos(math.radians(x))


def create_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def d3_grid_to_2d_image(grid, resolution):
    map_y = np.zeros([resolution*2, resolution*2])
    for i in range(grid.shape[0]):
        if map_y[int(grid[i][2])][int(grid[i][0])] == 0:
            map_y[int(grid[i][2])][int(grid[i][0])] = grid[i][1]
    return map_y


def save_image(color_map, file_str, set_name, number):
    im = Image.fromarray(color_map).convert('RGB')
    class_name = file_str.split('/')[0]
    class_id = file_str.split('/')[1]
    image_path = images_path + '/' + class_name + '/' + set_name
    create_folder(image_path)
    im.save(image_path + '/%s_%s.png' % (class_id, number))


def point2images(points, file_path, set_name, rate=56):
    for v in range(12):
        degree1 = 15
        degree2 = v * 30
        transform_x = np.asarray([[1, 0, 0],
                                  [0, cos(degree1), sin(degree1)],
                                  [0, -sin(degree1), cos(degree1)]])
        transform_y = np.asarray([[cos(degree2), 0, sin(degree2)],
                                  [0, 1, 0],
                                  [-sin(degree2), 0, cos(degree2)]])
        transform_xy = np.dot(transform_x, transform_y)
        view_point = np.dot(points, transform_xy)
        pixels = np.zeros(points.shape)
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                pixels[i][j] = rate + int(rate * view_point[i][j])
        number = v + 1
        image = d3_grid_to_2d_image(pixels, rate)
        save_image(image, file_path, set_name, number)


classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
              'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
              'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
              'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
              'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

if __name__ == '__main__':
    set_ = ['ply_data_train0.h5', 'ply_data_train1.h5', 'ply_data_train2.h5', 'ply_data_train3.h5',
            'ply_data_train4.h5', 'ply_data_test0.h5', 'ply_data_test1.h5']
    id_set = ['ply_data_train_0_id2file.json', 'ply_data_train_1_id2file.json', 'ply_data_train_2_id2file.json',
              'ply_data_train_3_id2file.json', 'ply_data_train_4_id2file.json', 'ply_data_test_0_id2file.json',
              'ply_data_test_1_id2file.json']
    h5_parent_path = 'E:\\CesareDou\\ModelNet40_ply_h5'
    images_path = 'E:\\CesareDou\\ModelNet40_images_2048_224_12views_45'
    rate = 112

    for indice in range(len(set_)):
        h5_path = h5_parent_path + '\\' + set_[indice]
        f = h5py.File(h5_path, 'r')  # 打开h5文件
        print(f.keys())  # 可以查看所有的主键
        datas = f['data'][:]  # 取出主键为data的所有的键值
        print(len(datas))
        # labels = f['label'][:]
        # normals = f['normal'][:]  # 为data中点的法线
        f.close()

        json_name = id_set[indice]
        with open(h5_parent_path + '/' + json_name) as load_f:
            id2file = json.load(load_f)

        if 'train' in set_[indice]:
            set_name = 'train'
        else:
            set_name = 'test'

        for num_model in range(len(id2file)):
            file_str = id2file[num_model]
            points = datas[num_model]
            point2images(points, file_str, set_name)
