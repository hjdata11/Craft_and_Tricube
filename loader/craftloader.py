import os.path as osp
import sys

import torch
import torch.utils.data as data
import cv2
import numpy as np
from utils.utils import WAPL_CLASSES
from kernel.kernel import craftmask
import os
import re
from utils.manipulation import generate_affinity_box
import json
from utils.mep import mep
from utils.imgproc import cvt2HeatmapImg



class ListDataset(data.Dataset):
    """VOC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, dataset, input_size, mode, split=None, transform=None, evaluation=False):
        self.root = root
        self.input_size = input_size
        self.dataset = dataset
        self.mode = mode
        self.split = split
        self.transform = transform
        self.evaluation = evaluation
        
        self.ids = list()

        if self.dataset == 'wapl':
            self.load_dataset(self.dataset)
            self.custom_class = WAPL_CLASSES
            self.num_classes = len(WAPL_CLASSES)

        else:
            raise "only support [GENERATED]"

        cv2.setNumThreads(0)

    def get_image_info_list(self, file_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                data_lines.extend(lines)
        return data_lines

    def load_dataset(self, dataset):
        self.target_transform = None
        
        custom_folder = osp.join(self.root, dataset)
        self.data_lines = self.get_image_info_list(os.path.join(custom_folder, self.mode + '_labels.txt'))
        self.data_idx_order_list = list(range(len(self.data_lines)))
        self.img_folder = osp.join(custom_folder, self.mode + '_images')

    def load_gt(self, label, txt):
        label = json.loads(label)
        nBox = len(label)
        bboxes = []
        for idx in range(0, nBox):
            box = label[idx]['points']
            box = [float(box[0][0]), float(box[0][1]), float(box[1][0]), float(box[1][1]), float(box[2][0]), float(box[2][1]), float(box[3][0]), float(box[3][1])]
            box = np.array(box, np.float32).reshape(4, 2)
            area, p0, p3, p2, p1, _, _ = mep(box)

            bbox = np.array([p0, p1, p2, p3])
            distance = 10000000
            index = 0
            for i in range(4):
                d = np.linalg.norm(box[0] - bbox[i])
                if distance > d:
                    index = i
                    distance = d
            new_box = []
            for i in range(index, index + 4):
                new_box.append(bbox[i % 4])
            new_box = np.array(new_box)
            bboxes.append(np.array(new_box))
    
        words = re.split(' \n|\n |\n| ', txt.strip())

        return bboxes, words
                
    def get_target(self, index):

        if self.dataset == 'wapl':
            file_idx = self.data_idx_order_list[index]
            data_line = self.data_lines[file_idx]
            try:
                data_line = data_line.decode('utf-8')
                substr = data_line.strip("\n").split("\t")
                file_name = substr[0].split("/")[-1]
                label = substr[1]
                txt = substr[2]
                image_path = os.path.join(self.img_folder, file_name)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image,  cv2.COLOR_BGR2RGB)
                character_bboxes, words = self.load_gt(label, txt)

                # _charbox = np.float32(_charbox)
            except Exception as e:
                self.logger.error(
                    "When parsing line {}, error happened with msg: {}".format(
                        data_line, e))
            

        else:
            raise "only support [Generated]"
                
        return image, character_bboxes, words


    def __getitem__(self, index):
        
        image, boxes, words = self.get_target(index)
        big_image, small_image, region_boxes = self.resize(image, boxes, big_side=self.input_size)  # Resize the image
        affinity_boxes = generate_affinity_box(region_boxes.copy(), words)

        
        region_scores = np.zeros((small_image.shape[0], small_image.shape[1]), dtype=np.float32)
        affinity_scores = np.zeros((small_image.shape[0], small_image.shape[1]), dtype=np.float32)

        region_boxes = np.reshape(region_boxes, (-1, 8))
        affinity_boxes = np.reshape(affinity_boxes, (-1, 8))
        
        for box in region_boxes:
            region_scores = craftmask(region_scores, box)

        for box in affinity_boxes:
            affinity_scores = craftmask(affinity_scores, box)

        big_image, region_scores, affinity_scores = self.transform(big_image, region_scores, affinity_scores)
        confidence_scores = np.ones((region_scores.shape[0], region_scores.shape[1]), dtype=np.float32)

        # mask_file = "./output/" + str(index) + '_mask_before.jpg'
        # cv2.imwrite(mask_file, cvt2HeatmapImg(region_scores))

        image = torch.from_numpy(big_image.astype(np.float32)).permute(2, 0, 1)
        region_scores_torch = torch.from_numpy(region_scores.astype(np.float32))
        affinity_scores_torch = torch.from_numpy(affinity_scores.astype(np.float32))
        confidence_scores_torch = torch.from_numpy(confidence_scores.astype(np.float32))

        return image, region_scores_torch, affinity_scores_torch, confidence_scores_torch
        
        
    def __len__(self):
        return len(self.data_idx_order_list)


    def resize(self, image, character, big_side):
        """
            Resizing the image while maintaining the aspect ratio and padding with average of the entire image to make the
            reshaped size = (side, side)
            :param image: np.array, dtype=np.uint8, shape=[height, width, 3]
            :param character: np.array, dtype=np.int32 or np.float32, shape = [2, 4, num_characters]
            :param side: new size to be reshaped to
            :return: resized_image, corresponding reshaped character bbox
        """

        height, width, channel = image.shape
        big_resize = (int(big_side), int(big_side))
        small_resize = (int(big_side//2), int(big_side//2))
        image = cv2.resize(image, big_resize)

        character = np.array(character)
        character[:, :, 0] = character[:, :, 0] * (small_resize[0] / width)
        character[:, :, 1] = character[:, :, 1] * (small_resize[1] / height)

        # big_image = np.ones([big_side, big_side, 3], dtype=np.float32)*255
        # h_pad, w_pad = (big_side-image.shape[0])//2, (big_side-image.shape[1])//2
        # big_image[h_pad: h_pad + image.shape[0], w_pad: w_pad + image.shape[1]] = image
        big_image = image.astype(np.uint8)

        small_image = cv2.resize(big_image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

        # character[:, :, 0] += (w_pad // 2)
        # character[:, :, 1] += (h_pad // 2)

        # character fit to small image
        return big_image, small_image, character
