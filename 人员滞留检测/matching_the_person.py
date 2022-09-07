import torch
from models import MobileNetV2IFN
import cv2
import numpy as np
from einops import repeat
from typing import List
from torchvision import transforms

class MatchPerson:
    def __init__(self,em_ckpt_file,device="cuda:{}".format(",".join([str(i) for i in range(torch.cuda.device_count())]))):
        # 创建mobilenet模型并加载参数。使用该模型提取特征
        self.device = torch.device(device)
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

    #计算相似度(欧几里得距离)
    def euclidean_dist(self,x, y):
        m, n = x.size(0), y.size(0)
        # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        # yy会在最后进行转置的操作
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        # torch.addmm(beta=1,?input,?alpha=1,?mat1,?mat2,?out=None)，这行表示的意思是dist - 2 * x * yT
        dist.addmm_(1, -2, x, y.t())
        # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
        dist = dist.clamp(min=1e-12).sqrt() #开方
        return dist
    #经测试dist_thres为0.08时图片目测相似度70-80%
    def matching_persons(self, img: np.ndarray, query_imgs: List[np.ndarray], dist_thres: float = 0.2):
        """
        :param img: 要匹配的图片
        :param query_imgs: 与img进行匹配的图片的列表
        :param dist_thres: 欧式距离的阈值
        :return:
        """
        gallery_imgs = [img]
        m, n = len(gallery_imgs), len(query_imgs)
        if m and n:
            _query_feats = self.more_encode(query_imgs) if n > 1 else self.encode(query_imgs[0])
            _gallery_feats = self.more_encode(gallery_imgs) if m > 1 else self.encode(gallery_imgs[0])
            distmat = torch.pow(_query_feats, 2).sum(dim=1, keepdim=True).expand(n, m) + \
                      torch.pow(_gallery_feats, 2).sum(dim=1, keepdim=True).expand(m, n).t()
            distmat.addmm_(_query_feats, _gallery_feats.t(), beta=1, alpha=-2).sqrt()
            distmat = distmat.cpu().numpy()  # <class 'tuple'>: (3, 12)
            targets = []#[(与img匹配的图片的index，向量距离值),]
            targets_distances = []#保存距离
            for i, v in enumerate(distmat):#小于设定的距离则认为是同一个目标
                if v < dist_thres:
                    targets.append(i)
                    targets_distances.append(v[0])
            #返回最相似的那张图片在query_imgs中的index
            if len(targets) == 0:
                return -1,100,[]
            elif len(targets) >= 1:
                most_likely_index = targets[np.argmin(np.array(targets_distances),axis=0)]
                return most_likely_index,targets_distances[np.argmin(np.array(targets_distances),axis=0)],targets









