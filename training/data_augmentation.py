# Data Augmentation
#
# Author: Xuan Li
# Time: 7/06/2020

import os
from os import listdir
from os.path import splitext
from glob import glob
from tqdm import tqdm
import imageio
import numpy as np
import albumentations as A

dir_img = './DATA/reviewed/Train/Images/'
dir_mask = './DATA/reviewed/Train/Masks/'
dir_out_img = './DATA/reviewed_augmented/Train/Images/'
dir_out_mask = './DATA/reviewed_augmented/Train/Masks/'
augNumPerImage = 2

# Customize this augmentation behavior in this function.
def get_augmenter():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            rotate=(-0.1, 0.1),
            translate_percent=(-0.025, 0.025),
            shear=(-0.025, 0.025),
            scale=(0.975, 1.025),
            p=1.0,
        ),
    ])

def get_item(idx):
    mask_file = glob(dir_mask + idx + '.*')
    img_file = glob(dir_img + idx + '.*')

    assert len(mask_file) == 1, \
        f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
    assert len(img_file) == 1, \
        f'Either no image or multiple images found for the ID {idx}: {img_file}'

    mask = imageio.imread(mask_file[0])
    img = imageio.imread(img_file[0])

    assert img.shape[:2] == mask.shape[:2], \
        f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

    return img, mask

if __name__ == "__main__":

    if not os.path.exists(dir_out_img):
        os.makedirs(dir_out_img)
    if not os.path.exists(dir_out_mask):
        os.makedirs(dir_out_mask)

    in_ids = [file for file in listdir(dir_img) if not file.startswith('.')]
    transform = get_augmenter()
    with tqdm(total=len(in_ids), desc=f'Image Processing', unit='img') as pbar:
        for id in in_ids:
            pathsplit = splitext(id)
            img, mask = get_item(pathsplit[0])
            for i in range(augNumPerImage):
                result = transform(image=img, mask=mask)
                imageio.imwrite("{}_aug_{}{}".format(dir_out_img + pathsplit[0], i, pathsplit[1]), result['image'])
                imageio.imwrite("{}_aug_{}{}".format(dir_out_mask + pathsplit[0], i, ".png"), result['mask'])
            imageio.imwrite("{}_aug_{}{}".format(dir_out_img + pathsplit[0], augNumPerImage, pathsplit[1]), img)
            imageio.imwrite("{}_aug_{}{}".format(dir_out_mask + pathsplit[0], augNumPerImage, ".png"), mask)
            pbar.update(1)
    
