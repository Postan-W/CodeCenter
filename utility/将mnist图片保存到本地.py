from keras.datasets import mnist
from PIL import Image
save_dir = "C:\\Users\\15216\\Pictures\\"
(x_train, y_train), (x_test, y_test) = mnist.load_data()
p1,p2,p3,p4,p5,p6 = x_train[1],x_train[12000],x_train[24000],x_train[36000],x_train[48000],x_train[59000]
# Image.fromarray(p1).save(save_dir+"0.jpg")
Image.fromarray(p2).save(save_dir+"7.jpg")
Image.fromarray(p3).save(save_dir+"8.jpg")
Image.fromarray(p4).save(save_dir+"9.jpg")
Image.fromarray(p5).save(save_dir+"4.jpg")
Image.fromarray(p6).save(save_dir+"6.jpg")
