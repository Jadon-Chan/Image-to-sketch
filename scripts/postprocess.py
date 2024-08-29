import cv2 as cv
from PIL import Image
import os
import numpy as np

def postprocess(image, results_path, init_image_name):
    image.save('tmp0.png')
    width, height = 300, 400
    image = image.resize(size=(width, height), resample=Image.Resampling.LANCZOS, reducing_gap=4.0)
    image = image.convert("L")
    image.save('tmp1.png')
    img = cv.imread('tmp1.png', cv.IMREAD_GRAYSCALE)
    img = cv.medianBlur(img,1)
    th = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2)
    cv.imwrite(f"{results_path}/{os.path.splitext(init_image_name)[0]}.png", th)

    with open(f"{results_path}/{os.path.splitext(init_image_name)[0]}.bin", 'wb') as file:
        result = np.array(th) 
        for i in range(200):
            print(result[i])
        result = result // 255
        padding = np.zeros((400, 4), dtype="int")
        result = np.concatenate((result, padding), 1)
        print(result.shape)
        result = result.flatten()
        print(len(result))
        for i in range(304*400//8):
            b = int(''.join(map(str, result[i:i+8])), 2)
            file.write(b.to_bytes())

if __name__ == "__main__":
    with Image.open("inits/girl.png") as image:
        postprocess(image, "results", "girl.png")
