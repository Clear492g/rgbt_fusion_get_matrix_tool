import json
import numpy as np
import pandas as pd
import cv2


def read_coordinate(filename):
    # 读取坐标
    with open(filename, 'r') as f:
        coordinates = json.load(f)
    return coordinates


def transformation_matrix(coordinates_other, coordinates_base):
    # 转换矩阵
    source, target = reshape_coordinate(coordinates_other, coordinates_base)
    transformation = cv2.estimateAffine2D(source, target, False)
    return transformation


def save_matrix2file(filename_with_path, matrix):
    print(matrix)
    # with open(folder + filename + '.npy', 'wb') as f:
    # np.save(f, matrix)
    pd.Series(list(matrix)).to_json(filename_with_path + '_matrix.json', orient='values')


def reshape_coordinate(coordinates_other, coordinates_base):
    source = []
    target = []
    for tag, coordinate in coordinates_other.items():
        source.append(coordinate)
        target.append(coordinates_base[tag])
    source = np.reshape(source, (-1, 2))
    target = np.reshape(target, (-1, 2))
    return source, target
