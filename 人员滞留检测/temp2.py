from procession import cv2_load_image
import queue
import cv2
import glob
q = queue.Queue()
# images = cv2_load_image("C:\\Users\\15216\\Desktop\\datasets\\超模脸GAN生成\\","generated_model-stylegan2\\")
image_path = glob.glob("C:\\Users\\15216\\Desktop\\datasets\\超模脸GAN生成\\generated_model-stylegan2\\*")
for image in image_path:
    q.put(cv2.imread(image))


