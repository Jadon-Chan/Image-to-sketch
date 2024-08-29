import cv2 as cv
from PIL import Image
import os
import numpy as np

img = cv.imread('tmp0.png', cv.IMREAD_GRAYSCALE)
# img = cv.medianBlur(img,3)
th = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2)
#cv.imwrite(f"{results_path}/{os.path.splitext(init_image_name)[0]}.png", th)
cv.imwrite("tmp1.png", th)
  
image = Image.fromarray(th) 

width, height = 300, 400
image = image.resize(size=(width, height), resample=Image.Resampling.LANCZOS, reducing_gap=3.0)
image = image.convert("L")
# image.save(f"{results_path}/{os.path.splitext(init_image_name)[0]}.png")


with open('tmp.bin', 'wb') as file:
    result = np.array(image) // 255
    print(result.shape)
    result = result.flatten()
    print(len(result))
    for i in range(300*400//8):
        b = int(''.join(map(str, result[i:i+8])), 2)
        file.write(b.to_bytes())
