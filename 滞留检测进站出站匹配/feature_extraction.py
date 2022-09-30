from torchvision import transforms
import torch
from models import MobileNetV2IFN
from einops import repeat
import numpy as np
from typing import List
import cv2

class Extractor:
    def __init__(self,em_ckpt_file="./weights/net_MobileNet_IFN.pth"):
        # 创建mobilenet模型并加载参数。使用该模型提取特征
        # gpu = "cuda:{}".format(",".join([str(i) for i in list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))))]))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.em_model = MobileNetV2IFN()
        self.em_model.load_weights(em_ckpt_file)
        self.em_model = self.em_model.half().to(self.device)
        self.em_model.eval()

        self.data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def process_img(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = self.data_transforms(image_rgb)
        img = img.unsqueeze(0)
        return img

    @staticmethod
    def horizontal_flip(img: torch.Tensor) -> torch.Tensor:
        """
        Performs horizontal flip of the input image.
        """
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    @torch.no_grad()
    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        :param frame: BGR image as a numpy array
        """
        first_batch = self.process_img(image)
        second_batch = self.horizontal_flip(first_batch)
        total_batch = torch.cat((first_batch,second_batch),0).half()
        total_batch = total_batch.to(self.device)
        output = self.em_model(total_batch).data.cpu()
        #embedding = output[0].unsqueeze(0)
        embedding = repeat(output[0], 'a -> b a', b=1)
        #print(type(embedding))
        embedding += output[1]
        fnorm = torch.norm(embedding, p=2, dim=1, keepdim=True)
        embedding = embedding.div(fnorm.expand_as(embedding))
        return embedding.float()

    #多张图片提取特征后的结果
    @torch.no_grad()
    def more_encode(self, images: List[np.ndarray]):
        #total_embedding = []
        first_batches = [self.process_img(image) for image in images]
        second_batches = [self.horizontal_flip(first_batch) for first_batch in first_batches]
        first_batches.extend(second_batches)
        new_batches = [first_batch.numpy()[0] for first_batch in first_batches]
        #total_batches = rearrange(new_batches, 'b l h d e -> b (l h) d e')
        #total_batches = torch.from_numpy(total_batches)
        total_batches = torch.FloatTensor(new_batches).half() #.squeeze(1)
        total_batches = total_batches.to(self.device)
        outputs = self.em_model(total_batches).data.cpu()
        outputs1, outputs2 = torch.chunk(outputs, chunks=2, dim=0)

        #nums = outputs1.shape[0]
        nums = outputs1.size(0)
        total_embedding = torch.zeros((nums,1280))#,dtype=np.float64)
        for i in range(nums):
            embedding = repeat(outputs1[i], 'a -> b a', b=1)
            embedding += outputs2[i]
            fnorm = torch.norm(embedding, p=2, dim=1, keepdim=True)
            embedding = embedding.div(fnorm.expand_as(embedding))
            total_embedding[i, :] = embedding

        return total_embedding.float()

    def extraction(self,images:List[np.ndarray]):
        n = len(images)
        return self.more_encode(images) if n > 1 else self.encode(images[0])