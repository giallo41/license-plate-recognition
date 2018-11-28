import os
import numpy as np
import cv2
import ast

IMG_DIR = './gen-data/images/'
MASK_DIR = './gen-data/mask/'
LABEL_DIR = './gen-data/labels/'

def read_img(filename, mode=cv2.IMREAD_COLOR, imshow=True,):
    
    img = cv2.imread(filename, mode)
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r,g,b])
    if imshow:
        plt.imshow(img2)
        plt.xticks([]) # x축 눈금
        plt.yticks([]) # y축 눈금
        plt.show()
    return img2

def save_img(filename, img):
    
    b, g, r = cv2.split(img)
    
    r, g, b = cv2.split(img)
    
    img2 = cv2.merge([b,g,r])
    cv2.imwrite(filename, np.ndarray.astype(img2, int))


def open_label_file (file_name):
    try: 
        with open(file_name) as f:
            file_contents = f.readlines()
        file_contents = [line.strip() for line in file_contents]
    except:
        print ('Error - Open File does not exsit :', file_name)
        return 

    obj = ast.literal_eval(file_contents[0])
    code = ast.literal_eval(file_contents[1])
    coord = ast.literal_eval(file_contents[2])
    m = ast.literal_eval(file_contents[3])
    
    return obj, code, coord, m


def get_affine_coord(M, coords):
    
    rtn_lst = []
    x_lst = []
    y_lst = []

    x, y, w, h = coords
    tmp_cd = [(x,y), (x+w,y), (x,y+h), (x+w,y+h)]
    
    M = np.array(M)
    
    for coord in tmp_cd: 
        x, y = coord
        tmp_mat = [x, y, 1]
        
        tmp_x = int(np.dot(M[0,:], tmp_mat))
        tmp_y = int(np.dot(M[1,:], tmp_mat))
        x_lst.append(tmp_x)
        y_lst.append(tmp_y)
        rtn_lst.append( [tmp_x , tmp_y ] )
    
    return np.max(x_lst), np.min(x_lst), np.max(y_lst), np.min(y_lst)


