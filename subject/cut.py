import argparse
import os
import random
import numpy as np
from PIL import Image

def check_size(image, n):
    row = image.shape[0]
    col = image.shape[1]
    
    if row % n == 0 and col % n == 0:
        return True
    else:
        return False    
    
    
def crop_image(image, n):
    row = image.shape[0]
    col = image.shape[1]
    
    re_row = row % n
    re_col = col % n
    
    image = image[n*re_row, n*re_col, :]
    
def mirror(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def flip(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def rotation(image):
    return image.transpose(Image.ROTATE_90)
        
def argumentation(image):
    rd = random.randint(1, 2)
    if rd > 1:
        image = mirror(image)
        
    rd = random.randint(1, 2)
    if rd > 1:
        image = flip(image)
        
    rd = random.randint(1, 2)
    if rd > 1:
        image = rotation(image)
        
    return image
    
def cut_image(image, n):
    list_image = []
    row = image.shape[0] % n
    col = image.shape[1] % n
    
    for i in range(row):
        for j in range(col):
            temp = image[i*2:i*2+1, j*2:j*2+1, :]
            temp = argumentation(temp)
            list_image.append(temp)
    
    return list_image 


if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input', type=str, default="input.png")
    parser.add_argument('--m', type=int, default=100)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--output', type=str, default="result")
    
    config = parser.parse_args()
    
    image = Image.open(config.input)
    n = config.n
    output_path = config.output
    
    if check_size == False:
        image = crop_image(image, n)
    
    if os.path.isdir(output_path):
        os.rmdir(output_path)
    os.mkdir(output_path)
    
    cut_image = cut_image(image, n)
    
    for i in cut_image:
        rd = random.randint(1000000000, 9999999999)
        i.save(output_path + "{}.png".format(rd))
    
        
        
    