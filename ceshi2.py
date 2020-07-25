from PIL import Image
import numpy as np
import os

ThresholdValue = 8


# def getFiles(source_path):
#     pass
#     return files

def is_Duplicated(img1_path, img2_path):
    return np.count_nonzero((
                                (Image.open(img1_path.strip()).resize((32, 32)).convert('L') >= np.mean(
                                    np.mean(Image.open(img1_path.strip()).resize((32, 32)).convert('L')))).astype(
                                    np.int8)) != (
                                        Image.open(img2_path.strip()).resize((32, 32)).convert('L') >= np.mean(
                                    np.mean(Image.open(img1_path.strip()).resize((32, 32)).convert('L')))).astype(np.int8)) < ThresholdValue

if __name__ == '__main__':
    img1_path = os.path.join(os.getcwd(), '1.jpg')
    img2_path = os.path.join(os.getcwd(), '2.jpg')
    print(is_Duplicated(img1_path, img2_path))
