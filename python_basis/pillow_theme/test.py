from PIL import Image
import numpy as np
image = Image.open("testrgb.jpg")
print("图像的尺寸是:",image.size)
image_numpy = np.array(image)
print("图像numpy数组的形状是:",image_numpy.shape)
#经过上面的两个输出对比，可以发现图像对象和numpy数组形式的宽高是相反的，具体顺序是怎样的需要看一下图像，缩减某个维度一般的量再看下图像，便可知
image_shrink = image.resize((int(image.size[0]/2),image.size[1]))
image_shrink.save("testrgb_shrinked.jpg")
"""
可以得到结论image对象的图像形状顺序是宽高，而numpy数组保存对的顺序是高宽通道
"""

image_gray = image.convert("L")
image_gray_numpy = np.array(image_gray)
print("转成灰度图像后的形状是:",image_gray.size)
print("转成灰度图像后numpy数组的形状是：",image_gray_numpy.shape)
image_gray.save("testrgb_gray.jpg")

image_gray_resized = image_gray.resize((32,32))
# image_gray_resized.show()
