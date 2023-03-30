import matplotlib.pyplot as plt
from PIL import Image
import json
import tkinter as tk
from tkinter import filedialog

import matrix_lib

global x, y, add_flag, pic_type, points_recorder
add_flag = True
points_recorder = {'other_img': {}, 'base_img': {}}


def on_press_other(event):
    global add_flag, x, y, pic_type
    pic_type = 'other_img'
    x = int(event.xdata)
    y = int(event.ydata)
    add_flag = False


def on_press_base(event):
    global add_flag, x, y, pic_type
    pic_type = 'base_img'
    x = int(event.xdata)
    y = int(event.ydata)
    add_flag = False


def press_keyboard(event):
    global add_flag, x, y, points_recorder, pic_type
    if len(event.key) == 1:
        print(pic_type + ' ' + 'add success ' + event.key)
        points_recorder[pic_type][event.key] = [x, y]
        add_point(event, x, y, event.key)
        add_flag = True
    elif event.key == 'escape':
        points_recode_finished()


def points_recode_finished():
    print('关键点记录完成。')

    filename_with_path_without_suffix = other_img_path.split('.')[0]
    save_points_recorder_to_file(filename_with_path_without_suffix)

    coordinates = matrix_lib.read_coordinate(filename_with_path_without_suffix + '.json')
    coordinates_other = coordinates['other_img']
    coordinates_base = coordinates['base_img']
    trans_matrix = matrix_lib.transformation_matrix(coordinates_other, coordinates_base)
    matrix_lib.save_matrix2file(filename_with_path_without_suffix, trans_matrix[0])
    exit(0)


def add_point(event, x_coordinate, y_coordinate, tag=None):
    # 记录新图片并显示点的位置，以判断是否需要添加点
    plt.plot(x_coordinate, y_coordinate, 'o', color='r')
    if tag is not None:
        plt.annotate(tag, (x_coordinate, y_coordinate), color='g')
    event.inaxes.figure.canvas.draw()  # 刷新canvas


def save_points_recorder_to_file(filename):
    global points_recorder
    # 创建一个文件并将坐标保存为json格式
    # suffix = filename_dc.split('_')[0]
    with open(filename + '.json', 'w') as f:
        json.dump(points_recorder, f)


if __name__ == '__main__':
    print('成像传感器标定工具V1.0.0')
    print('Powered By Clear')
    print('******************')
    root = tk.Tk()
    root.withdraw()
    print('选择基底图像...')
    base_img_path = filedialog.askopenfilename(title='选择基底图像...')
    print('选择融合图像...')
    other_img_path = filedialog.askopenfilename(title='选择融合图像...')

    fig = plt.figure()
    img_input = Image.open(other_img_path)
    fig.canvas.mpl_connect('button_press_event', on_press_other)
    fig.canvas.mpl_connect('key_press_event', press_keyboard)
    plt.imshow(img_input, animated=True)

    fig2 = plt.figure()
    img_base = Image.open(base_img_path)
    fig2.canvas.mpl_connect('button_press_event', on_press_base)
    fig2.canvas.mpl_connect('key_press_event', press_keyboard)
    plt.imshow(img_base)

    plt.show()

    print('使用指针点击像素后按下单字符按键,结束使用esc')
