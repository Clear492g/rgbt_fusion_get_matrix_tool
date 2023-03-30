import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def load_transformation_matrix(filename):
    with open(filename, 'rb') as f:
        matrix = json.load(f)
    return np.array(matrix)


def read_coordinate(filename):
    with open(filename, 'r') as f:
        coordinates = json.load(f)
    return coordinates


def predict_base_by_other(coordinates_other, matrix):
    coordinates_predict = {}
    trans_matrix = matrix.T  # reshape to (3, 2)
    for tag, coordinate in coordinates_other.items():
        coordinate.append(1)
        coordinate = np.reshape(coordinate, (1, 3))
        predict_result = np.dot(coordinate, trans_matrix).tolist()[0]
        coordinates_predict[tag] = list(map(int, predict_result))  # cast float type to int
    return coordinates_predict


def maximum_internal_rectangle(img_mask): # 计算内接矩形
    img = img_mask.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contour = contours[0].reshape(len(contours[0]), 2)

    rect = []

    for i in range(len(contour)):
        x1, y1 = contour[i]
        for j in range(len(contour)):
            x2, y2 = contour[j]
            area = abs(y2 - y1) * abs(x2 - x1)
            rect.append(((x1, y1), (x2, y2), area))

    all_rect = sorted(rect, key=lambda x: x[2], reverse=True)

    if all_rect:
        best_rect_found = False
        index_rect = 0
        nb_rect = len(all_rect)

        while not best_rect_found and index_rect < nb_rect:

            rect = all_rect[index_rect]
            (x1, y1) = rect[0]
            (x2, y2) = rect[1]

            valid_rect = True

            x = min(x1, x2)
            while x < max(x1, x2) + 1 and valid_rect:
                if any(img[y1, x]) == 0 or any(img[y2, x]) == 0:
                    valid_rect = False
                x += 1

            y = min(y1, y2)
            while y < max(y1, y2) + 1 and valid_rect:
                if any(img[y, x1]) == 0 or any(img[y, x2]) == 0:
                    valid_rect = False
                y += 1

            if valid_rect:
                best_rect_found = True

            index_rect += 1

        if best_rect_found:
            # 如果要在灰度图img_gray上画矩形，请用黑色画（0,0,0）
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            print((x1, y1), (x2, y2))
            cv2.imshow("rec", img)
            cv2.waitKey(0)


        else:
            print("No rectangle fitting into the area")

    else:
        print("No rectangle found")



if __name__ == '__main__':

    matrix = load_transformation_matrix('pic/thermal_matrix.json')

    coordinates = read_coordinate('pic/thermal.json')
    coordinates_other = coordinates['other_img']
    coordinates_base = coordinates['base_img']
    coordinates_predict = predict_base_by_other(coordinates_other, matrix)
    print(coordinates_other)
    print(coordinates_predict)

    img_other = Image.open('pic/thermal.png')
    img_base = Image.open('pic/rgb.png')


    fig = plt.figure()

    ax0 = fig.add_subplot(132)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(133)

    for tag, coordinate in coordinates_other.items():
        ax1.scatter(coordinate[0], coordinate[1])
        ax1.annotate(tag, (coordinate[0], coordinate[1]), color='g')
    for tag, coordinate in coordinates_base.items():
        ax2.scatter(coordinate[0], coordinate[1])
        ax2.annotate(tag, (coordinate[0], coordinate[1]), color='g')
    for tag, coordinate in coordinates_predict.items():
        ax2.scatter(coordinate[0], coordinate[1], color='b')

    img = cv2.imread('pic/thermal.png')
    img2 = cv2.imread('pic/rgb.png')
    rows, cols, channel = img2.shape
    dst = cv2.warpAffine(img, M=matrix, dsize=(cols, rows))

    ax0.set_title('clear')
    ax0.imshow(dst)


    ax1.set_title('other_pic')
    ax1.imshow(img_other)
    ax2.set_title('base_pic')
    ax2.imshow(img_base)

    plt.show()

    maximum_internal_rectangle(dst)
