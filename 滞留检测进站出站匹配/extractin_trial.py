import cv2

from feature_extraction import Extractor
import glob
extractor = Extractor()
images_path = glob.glob("./trial/person_detected/*")
images = []
for image_path in images_path:
    images.append(cv2.imread(image_path))

images_feature = extractor.extraction(images)#每个特征的长度是1280,类型是torch.tensor
print(type(images_feature))
images_feature = list(images_feature.numpy())
print(images_feature[0])

