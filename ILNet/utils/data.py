import jittor as jt
from jittor.dataset import Dataset
import jittor.transform as transform
from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import random
import numpy as np

class SirstDataset(Dataset):
    def __init__(self, args, mode='train'):
        super().__init__() # 必须调用父类初始化
        base_dir = 'datasets/SIRST'

        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'val':
            txtfile = 'test.txt'

        self.list_dir = osp.join(base_dir, 'idx_427', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.img_size = args.img_size
        
        # Jittor transform 替代 torchvision
        self.img_transform = transform.Compose([
            transform.ToTensor(),
            transform.ImageNormalize([0.35619214, 0.35575104, 0.35673013], 
                                     [0.2614548, 0.26135704, 0.26168558]),
        ])
        self.mask_transform = transform.ToTensor()
        
        # 设置 Jittor Dataset 的 batch_size 等参数（也可以在外部实例化时设置）
        self.set_attrs(batch_size=args.batch_size, shuffle=(mode=='train'))

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + '_pixels0.png')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
        
        # 转换为 Jittor 能够处理的 numpy 数组或直接转换
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        img_size = self.img_size
        long_size = random.randint(int(self.img_size * 0.5), int(self.img_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        if short_size < img_size:
            padh = img_size - oh if oh < img_size else 0
            padw = img_size - ow if ow < img_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        
        w, h = img.size
        x1 = random.randint(0, w - img_size)
        y1 = random.randint(0, h - img_size)
        img = img.crop((x1, y1, x1 + img_size, y1 + img_size))
        mask = mask.crop((x1, y1, x1 + img_size, y1 + img_size))

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return img, mask

    def _testval_sync_transform(self, img, mask):
        img_size = self.img_size
        img = img.resize((img_size, img_size), Image.BILINEAR)
        mask = mask.resize((img_size, img_size), Image.NEAREST)
        return img, mask

class IRSTD1K_Dataset(SirstDataset):
    def __init__(self, args, mode='train'):
        # IRSTD1K 逻辑与 SIRST 极其相似，只需修改路径和 Normalize 参数
        super(SirstDataset, self).__init__() 
        base_dir = 'datasets/IRSTD-1k'

        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'val':
            txtfile = 'test.txt'

        self.list_dir = osp.join(base_dir, txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.img_size = args.img_size
        self.img_transform = transform.Compose([
            transform.ToTensor(),
            transform.ImageNormalize([0.28450727, 0.28450724, 0.28450724], 
                                     [0.22880708, 0.22880709, 0.22880709]),
        ])
        self.mask_transform = transform.ToTensor()
        self.set_attrs(batch_size=args.batch_size, shuffle=(mode=='train'))

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + '.png') # 注意这里后缀与 SIRST 不同

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
        
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        return img, mask

def get_mean_std(data_set):
    # Jittor Dataset 遍历方式
    sum_of_pixels = np.zeros(3)
    sum_of_square_error = np.zeros(3)
    n = 0
    
    for X, _ in data_set:
        # X: [C, H, W]
        sum_of_pixels += X.sum(axis=(1, 2))
        n += X.shape[1] * X.shape[2]
        
    _mean = sum_of_pixels / n
    
    for X, _ in data_set:
        for d in range(3):
            sum_of_square_error[d] += np.sum((X[d, :, :] - _mean[d])**2)
            
    _std = np.sqrt(sum_of_square_error / n)
    return _mean.tolist(), _std.tolist()