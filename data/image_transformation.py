import cv2
import os
import numpy as np
import random
 
def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    rotated_img = cv2.warpAffine(image, M, (w, h))

    return rotated_img

def move(image, m_h, m_w):
    (h, w) = image.shape[:2]
    M = np.float32([[1,0,m_w],[0,1,m_h]])
    moved_img = cv2.warpAffine(image, M, (h,w))

    return moved_img


def tranform(img):
    ## 对reference进行非对齐变换
    angle = random.uniform(-10,10)
    shift_h = random.uniform(-10,10)
    shift_w = random.uniform(-10,10)
    rand_data = random.random()
    rand_rotation = random.random()
    rand_move_h = random.random()
    rand_move_w = random.random()
    if rand_data<=0.3:
        img = img
    else:
        if rand_rotation < 0.5:
            img = rotate_bound(img, angle)
        if rand_move_h <0.5:
            img = move(img, 0, shift_h)
        if rand_move_w < 0.5:
            img = move(img, shift_w, 0)

    return img


img_file = 'input file path'
save_file = 'save file path'
file_name_list = sorted(os.listdir(img_file))

for name in file_name_list:
    img = cv2.imread(os.path.join(img_file, name), cv2.IMREAD_UNCHANGED)
    img = tranform(img)
    cv2.imwrite(os.path.join(save_file, name), img)
    print(name)


    
