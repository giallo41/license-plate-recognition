import math
import os
import random
import sys

import cv2
import numpy as np
import pandas as pd

import skimage.io
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

## Plate Color combination of Korean 
## List of ( Character Color , Plate Color )
PLATE_COLOR = [ 
    ( 'black', 'yellow'),
    ('black', 'white'),
    ('blue', 'yellow'),
    ('white', 'green'),
    ('white', 'blue')
]


DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
KOR_LETTERS = "가나다라마바사하허호"
CHARS = LETTERS + DIGITS

FONT_DIR = "./font"
FONT_HEIGHT = 32  # Pixel size to which the chars are resized

OUTPUT_SHAPE = (128, 128)

CHARS = CHARS + " "

def make_char_ims(font_path, output_height):
    font_size = output_height * 4

    font = ImageFont.truetype(font_path, font_size)

    height = max(font.getsize(c)[1] for c in CHARS)

    for c in CHARS:
        width = font.getsize(c)[0]
        im = Image.new("RGBA", (width, height), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, np.array(im)[:, :, 0].astype(np.float32) / 255.
        

def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = np.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = np.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = np.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M

def pick_colors():
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color


def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = np.array([[from_shape[1], from_shape[0]]]).T
    to_size = np.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = np.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = np.array(np.max(M * corners, axis=1) -
                              np.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= np.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (np.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if np.any(trans < -0.5) or np.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = np.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds

def generate_code():
    return "{}{} {} {}{}{}{}".format(
        random.choice(DIGITS),
        random.choice(DIGITS),
        random.choice(LETTERS),
        random.choice(DIGITS),
        random.choice(DIGITS),
        random.choice(DIGITS),
        random.choice(DIGITS))


def rounded_rect(shape, radius):
    out = np.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    return out


def generate_plate(font_height, char_ims):
    h_padding = random.uniform(0.2, 0.4) * font_height
    v_padding = random.uniform(0.1, 0.3) * font_height
    spacing = font_height * random.uniform(-0.05, 0.05)
    radius = 1 + int(font_height * 0.1 * random.random())

    code = generate_code()
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                 int(text_width + h_padding * 2))

    text_color, plate_color = pick_colors()
    
    text_mask = np.zeros(out_shape)
    
    x = h_padding
    y = v_padding 
    for c in code:
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

    plate = (np.ones(out_shape) * plate_color * (1. - text_mask) +
             np.ones(out_shape) * text_color * text_mask)

    return plate, rounded_rect(out_shape, radius), code.replace(" ", "")


def generate_bg(num_bg_images):
    found = False
    while not found:
        fname = "./bg-image/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
        bg = skimage.io.imread(fname)/255.
        #bg = cv2.imread(fname, cv2.CV_LOAD_IMAGE_GRAYSCALE) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

    return bg


def generate_im(char_ims, num_bg_images):
    bg = generate_bg(num_bg_images)

    plate, plate_mask, code = generate_plate(FONT_HEIGHT, char_ims)
    
    M, out_of_bounds = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=0.4,
                            max_scale=0.675,
                            rotation_variation=1.0,
                            scale_variation=1.5,
                            translation_variation=1.2)
    
    coords = [(0,0), (plate.shape[1], 0), (0, plate.shape[0]), (plate.shape[1], plate.shape[0])]
    
    affine_coord, out_of_bound = get_affine_coord(M, coords, boundary = (128, 128))
    
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))
    
    plate = np.expand_dims(plate, axis=-1)
    plate_mask = np.expand_dims(plate_mask, axis=-1)
    
    out = plate * plate_mask + bg * (1.0 - plate_mask)

    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    out += np.random.normal(scale=0.05, size=out.shape)
    out = np.clip(out, 0., 1.)

    return out, plate_mask, code, np.array(affine_coord), out_of_bound


def load_fonts(folder_path):
    font_char_ims = {}
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]
    for font in fonts:
        font_char_ims[font] = dict(make_char_ims(os.path.join(folder_path,
                                                              font),
                                                 FONT_HEIGHT))
    return fonts, font_char_ims

def get_affine_coord(M, coords, boundary = (128, 128)):
    
    rtn_lst = []
    x_lst = []
    y_lst = []
    out_of_bound = False
    
    for coord in coords: 
        x, y = coord
        tmp_mat = [x, y, 1]
        
        tmp_x = int(np.dot(M[0,:], tmp_mat))
        tmp_y = int(np.dot(M[1,:], tmp_mat))
        x_lst.append(tmp_x)
        y_lst.append(tmp_y)
        rtn_lst.append( [tmp_x , tmp_y ] )
    
    b_width, b_height = boundary
    
    rtn_lst = np.array(rtn_lst )
    
    if np.sum(rtn_lst[:,0]<0) > 0 or np.sum(rtn_lst[:,0]>b_width) > 0:
        out_of_bound = True
    elif np.sum(rtn_lst[:,1]<0) > 0 or np.sum(rtn_lst[:,1]>b_width) > 0:
        out_of_bound = True
    
    return np.array(rtn_lst ), out_of_bound


def main():
    
    file_names = []
    coords = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    x4 = []
    y4 = []

    try :
        os.mkdir("train-data-gen")
    except:
        None

    try :
        os.mkdir("./train-data-gen/images/")
    except:
        None

    try :
        os.mkdir("./train-data-gen/mask/")
    except:
        None

    try :
        os.mkdir("./train-data-gen/labels/")
    except:
        None

    img_dir = './train-data-gen/images/'
    mask_dir = './train-data-gen/mask/'
    label_dir = './train-data-gen/labels/'

    fonts, font_char_ims = load_fonts(FONT_DIR)

    idx = 0
    for i in range (10):
        out_im, plate_mask, code, affine_coord, out_of_bound = generate_im(font_char_ims[random.choice(fonts)], 1000)
        if out_of_bound :
            continue
        else:
            file_name = "{:08d}.png".format(idx)
            file_names.append(file_name)
            coords.append(code)
            x1.append(affine_coord[0,0])
            y1.append(affine_coord[0,1])
            x2.append(affine_coord[1,0])
            y2.append(affine_coord[1,1])
            x3.append(affine_coord[2,0])
            y3.append(affine_coord[2,1])
            x4.append(affine_coord[3,0])
            y4.append(affine_coord[3,1])
            idx += 1
            if i%3 == 0:
                print (file_name)
            cv2.imwrite(os.path.join(img_dir, file_name), out_im*255.)
            cv2.imwrite(os.path.join(mask_dir, file_name), plate_mask*255.)

    df = pd.DataFrame( index = range(0,len(file_names)))
    df['file_name'] = file_names
    df['coord'] = coords
    df['x1'] = x1
    df['y1'] = y1
    df['x2'] = x2
    df['y2'] = y2
    df['x3'] = x3
    df['y3'] = y3
    df['x4'] = x4
    df['y4'] = y4

    df.to_csv(os.path.join(label_dir, 'labels.csv'), index=False)
    print ('Datasets Generated !!')



if __name__ == '__main__':
    main()