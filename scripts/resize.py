from PIL import Image
import numpy as np

# with Image.open("inits/image.png") as im:
#     width, height = 1080, 1920
#     im = im.resize((width, height), resample=Image.Resampling.LANCZOS)

# im.save("inits/image_resize.png")

with Image.open("inits/image.png") as im:
    im = np.array(im)
    print(im.shape)
    print(im.dtype)
    new_im = np.zeros((240, 240, 3), dtype=int)
    for x in range(0, 240):
        for y in range(0, 240):
            x_ = 480 + 4*x
            y_ = 60 + 4*y
            sum = np.zeros((3,), dtype=int)
            for i in range(0, 4):
                for j in range(0, 4):
                    sum += im[x_+i][y_+j]
            sum = sum // 16
            new_im[x][y] = sum
    new_im = Image.fromarray(np.uint8(new_im))
    new_im.save("inits/image_recolor.png")
