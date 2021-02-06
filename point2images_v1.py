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
    map_x = np.zeros([resolution*2, resolution*2])
    map_y = np.zeros([resolution*2, resolution*2])
    map_z = np.zeros([resolution*2, resolution*2])
    for i in range(grid.shape[0]):
        if map_x[int(grid[i][2])][int(grid[i][1])] == 0:
            map_x[int(grid[i][2])][int(grid[i][1])] = grid[i][0]

        if map_y[int(grid[i][2])][int(grid[i][0])] == 0:
            map_y[int(grid[i][2])][int(grid[i][0])] = grid[i][1]

        if map_z[int(grid[i][0])][int(grid[i][1])] == 0:
            map_z[int(grid[i][0])][int(grid[i][1])] = grid[i][2]

    return map_x, map_y, map_z


def d3_grid_to_2d_image_(grid, resolution):
    grid = -grid[:]
    map_x = np.zeros([resolution * 2, resolution * 2])
    map_y = np.zeros([resolution * 2, resolution * 2])
    map_z = np.zeros([resolution * 2, resolution * 2])
    for i in range(grid.shape[0]):
        if map_x[int(grid[i][2])][int(grid[i][1])] == 0:
            map_x[int(grid[i][2])][int(grid[i][1])] = grid[i][0]
        else:
            if map_x[int(grid[i][2])][int(grid[i][1])] < grid[i][0]:
                map_x[int(grid[i][2])][int(grid[i][1])] = grid[i][0]

        if map_y[int(grid[i][2])][int(grid[i][0])] == 0:
            map_y[int(grid[i][2])][int(grid[i][0])] = grid[i][1]
        else:
            if map_x[int(grid[i][2])][int(grid[i][0])] < grid[i][1]:
                map_x[int(grid[i][2])][int(grid[i][0])] = grid[i][1]

        if map_z[int(grid[i][0])][int(grid[i][1])] == 0:
            map_z[int(grid[i][0])][int(grid[i][1])] = grid[i][2]
        else:
            if map_x[int(grid[i][0])][int(grid[i][1])] < grid[i][2]:
                map_x[int(grid[i][0])][int(grid[i][1])] = grid[i][2]

    map_x = - map_x[:]
    map_y = - map_y[:]
    map_z = - map_z[:]
    return map_x, map_y, map_z


def save_image(color_map, file_str, set_name, number):
    im = Image.fromarray(color_map).convert('RGB')
    class_name = file_str.split('/')[0]
    class_id = file_str.split('/')[1]
    image_path = images_path + '/' + class_name + '/' + set_name
    create_folder(image_path)
    im.save(image_path + '/%s_%s.png' % (class_id, number))


def point2images(points, file_path, set_name, rate=112):
    pixels = np.zeros(points.shape)
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            pixels[i][j] = rate + int(rate * points[i][j])
    # print(points[0], pixels[0])
    color_maps = [[], [], []]
    color_maps[0], color_maps[1], color_maps[2] = d3_grid_to_2d_image(pixels, rate)
    # color_maps = [[], [], [], [], [], []]
    # color_maps[0], color_maps[2], color_maps[4] = d3_grid_to_2d_image(pixels, rate)
    # color_maps[1], color_maps[3], color_maps[5] = d3_grid_to_2d_image_(pixels, rate)
    # transform_x = np.asarray([[1, 0, 0],
    #                           [0, cos(45), sin(45)],
    #                           [0, -sin(45), cos(45)]])
    # transform_y = np.asarray([[cos(45), 0, sin(45)],
    #                           [0, 1, 0],
    #                           [-sin(45), 0, cos(45)]])
    # transform_z = np.asarray([[cos(45), -sin(45), 0],
    #                           [sin(45), cos(45), 0],
    #                           [0, 0, 1]])
    # transforms = np.dot(np.dot(transform_x, transform_y), transform_z)
    # points_ = np.dot(points, transforms)
    # pixels_ = np.zeros(points.shape)
    # for i in range(points_.shape[0]):
    #     for j in range(points_.shape[1]):
    #         pixels_[i][j] = rate + int(rate * points_[i][j])
    # color_maps[6], color_maps[8], color_maps[10] = d3_grid_to_2d_image(pixels_, rate)
    # color_maps[7], color_maps[9], color_maps[11] = d3_grid_to_2d_image_(pixels_, rate)
    for v in range(3):
        number = v + 1
        save_image(color_maps[v], file_path, set_name, number)


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
    images_path = 'E:\\CesareDou\\ModelNet40_images_2048_224_3views'
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
        with open(h5_parent_path + '\\' + json_name) as load_f:
            id2file = json.load(load_f)

        if 'train' in set_[indice]:
            set_name = 'train'
        else:
            set_name = 'test'

        for num_model in range(len(id2file)):
            file_str = id2file[num_model]
            points = datas[num_model]
            point2images(points, file_str, set_name, rate)
