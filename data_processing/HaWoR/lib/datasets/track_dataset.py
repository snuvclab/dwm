import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, ToTensor, Compose
import numpy as np
import cv2

from lib.core import constants
from lib.utils.imutils import crop, boxes_2_cs


class TrackDatasetEval(Dataset):
    """
    Track Dataset Class - Load images/crops of the tracked boxes.
    """
    def __init__(self, imgfiles, boxes, 
                 crop_size=256, dilate=1.0,
                img_focal=None, img_center=None, normalization=True,
                item_idx=0, do_flip=False):
        super(TrackDatasetEval, self).__init__()

        self.imgfiles = imgfiles
        self.crop_size = crop_size
        self.normalization = normalization
        self.normalize_img = Compose([
                            ToTensor(),
                            Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
                        ])

        self.boxes = boxes
        self.box_dilate = dilate
        self.centers, self.scales = boxes_2_cs(boxes)

        self.img_focal = img_focal
        self.img_center = img_center
        self.item_idx = item_idx
        self.do_flip = do_flip

    def __len__(self):
        return len(self.imgfiles)
    
    
    def __getitem__(self, index):
        item = {}
        imgfile = self.imgfiles[index]
        scale = self.scales[index] * self.box_dilate
        center = self.centers[index]

        img_focal = self.img_focal
        img_center = self.img_center

        img = cv2.imread(imgfile)[:,:,::-1]
        if self.do_flip:
            img = img[:, ::-1, :]
            img_width = img.shape[1]
            center[0] = img_width - center[0] - 1
        img_crop = crop(img, center, scale, 
                        [self.crop_size, self.crop_size], 
                        rot=0).astype('uint8')
        # cv2.imwrite('debug_crop.png', img_crop[:,:,::-1])
        
        if self.normalization:
            img_crop = self.normalize_img(img_crop)
        else:
            img_crop = torch.from_numpy(img_crop)
        item['img'] = img_crop
        
        if self.do_flip:
            # center[0] = img_width - center[0] - 1 
            item['do_flip'] = torch.tensor(1).float()
        item['img_idx'] = torch.tensor(index).long()
        item['scale'] = torch.tensor(scale).float()
        item['center'] = torch.tensor(center).float()
        item['img_focal'] = torch.tensor(img_focal).float()
        item['img_center'] = torch.tensor(img_center).float()
        

        return item

