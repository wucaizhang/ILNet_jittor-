import jittor as jt
import numpy as np
from skimage import measure

class IoUMetric:
    def __init__(self):
        self.reset()

    def update(self, pred, labels):
        # 对齐作者：直接对 Logits 进行 > 0 处理
        if isinstance(pred, jt.Var): pred = pred.numpy()
        if isinstance(labels, jt.Var): labels = labels.numpy()

        predict = (pred > 0).astype('int64')
        target = (labels > 0).astype('int64')
        
        # 像素级 TP, T, Union
        pixel_correct = np.sum((predict == target) * (target > 0))
        pixel_labeled = np.sum(target > 0)
        
        intersection = predict * (predict == target)
        area_inter, _ = np.histogram(intersection, bins=1, range=(1, 1))
        area_pred, _ = np.histogram(predict, bins=1, range=(1, 1))
        area_lab, _ = np.histogram(target, bins=1, range=(1, 1))
        area_union = area_pred + area_lab - area_inter

        self.total_correct += pixel_correct
        self.total_label += pixel_labeled
        self.total_inter += area_inter
        self.total_union += area_union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        return pixAcc, IoU.mean()

    def reset(self):
        self.total_inter = 0; self.total_union = 0
        self.total_correct = 0; self.total_label = 0

class nIoUMetric:
    def __init__(self, nclass=1, score_thresh=0.5):
        self.score_thresh = score_thresh
        self.reset()

    def update(self, preds, labels):
        # 对齐作者：nIoU 必须先过 Sigmoid 再判断 > 0.5
        if isinstance(preds, jt.Var):
            preds = jt.sigmoid(preds).numpy()
        if isinstance(labels, jt.Var):
            labels = labels.numpy()

        for b in range(preds.shape[0]):
            predict = (preds[b] > self.score_thresh).astype('int64')
            target = (labels[b] > 0).astype('int64')
            
            inter = np.sum(predict * (predict == target))
            area_pred = np.sum(predict)
            area_lab = np.sum(target)
            union = area_pred + area_lab - inter
            
            self.total_inter = np.append(self.total_inter, inter)
            self.total_union = np.append(self.total_union, union)

    def get(self):
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        return None, IoU.mean()

    def reset(self):
        self.total_inter = np.array([]); self.total_union = np.array([])

class PD_FA:
    def __init__(self, img_size):
        self.img_size = img_size
        self.reset()

    def update(self, preds, labels):
        if isinstance(preds, jt.Var): preds = preds.numpy()
        if isinstance(labels, jt.Var): labels = labels.numpy()

        # 对齐作者：Pd/FA 使用的是原始输出 > 0
        for b in range(preds.shape[0]):
            pred_b = (preds[b, 0] > 0).astype('int64')
            label_b = (labels[b, 0] > 0).astype('int64')

            # connectivity=2 是 8 连通，必须保持一致
            pred_label = measure.label(pred_b, connectivity=2)
            coord_image = measure.regionprops(pred_label)
            true_label = measure.label(label_b, connectivity=2)
            coord_label = measure.regionprops(true_label)

            self.target += len(coord_label)
            
            image_area_total = [prop.area for prop in coord_image]
            image_area_match = []
            matched_targets = 0

            for i in range(len(coord_label)):
                centroid_label = np.array(coord_label[i].centroid)
                for m in range(len(coord_image)):
                    centroid_image = np.array(coord_image[m].centroid)
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    if distance < 3:
                        image_area_match.append(coord_image[m].area)
                        matched_targets += 1
                        # 匹配后删除，防止重复计数
                        del coord_image[m]
                        break
            
            self.PD += matched_targets
            # FA 计算：预测中未匹配上的面积
            dismatch_area = sum(image_area_total) - sum(image_area_match)
            self.FA += dismatch_area

    def get(self, img_num):
        Final_FA = self.FA / ((self.img_size * self.img_size) * img_num)
        Final_PD = self.PD / (self.target + np.spacing(1))
        return Final_FA, Final_PD

    def reset(self):
        self.FA = 0; self.PD = 0; self.target = 0