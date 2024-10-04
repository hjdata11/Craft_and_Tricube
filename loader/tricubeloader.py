import os.path as osp
import sys
from tkinter import W

from utils.utils import random_scale
import torch
import torch.utils.data as data
import cv2
import numpy as np
from utils.utils import GENERATED_CLASSES
from kernel.kernel import tricubemask
import re
from utils.manipulation import generate_affinity_box
import xml.etree.ElementTree as ET


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
        
        if self.dataset == 'generated':
            self.load_dataset(self.dataset)
            self.custom_class = GENERATED_CLASSES
            self.num_classes = len(GENERATED_CLASSES)

        else:
            raise "only support [GENERATED]"

        cv2.setNumThreads(0)

    def load_dataset(self, dataset):
        self.target_transform = None
        
        self._anno_path = osp.join(self.root, dataset, self.mode, 'labels', '%s.xml')
        self._img_path = osp.join(self.root, dataset, self.mode, 'images', '%s.jpg')
        dataset_list = osp.join(self.root, dataset, self.mode, "img_list.txt")

        dataset_list = open(dataset_list, "r")
        
        for line in dataset_list.read().splitlines():
            self.ids.append(line)
        
        self.ids = sorted(self.ids)

    def distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
    
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
    def get_target(self, img_id):

        if self.dataset == 'generated':
            img_path = self._img_path % (img_id)
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
            height, width = img.shape[0], img.shape[1]
            # img, height, width = padding_img(img, height, width)

            size = img.shape[0]
        
            if 'test' in self.mode:
                return [], img_path, img

            tree = ET.parse(self._anno_path % img_id)
            root = tree.getroot()

            boxes = []
            labels = []
            txt = root.findtext('txt')
            words = re.split(' \n|\n |\n| ', txt.strip())

            for object in root.iter('object'):
                xmin = object.find('bndbox').findtext('xmin')
                ymin = object.find('bndbox').findtext('ymin')
                xmax = object.find('bndbox').findtext('xmax')
                ymax = object.find('bndbox').findtext('ymax')
                text = str(object.findtext('name'))

                box = [float(xmin), float(ymin), float(xmax), float(ymin), float(xmax), float(ymax), float(xmin), float(ymax)]
                box = np.array(box, np.float32).reshape(4, 2)
                boxes.append(box)
                labels.append(0)
        else:
            raise "only support [Generated]"
                
        return boxes, labels, words, img_path, img


    def __getitem__(self, index):

        data_id = self.ids[index]
        
        boxes, labels, words, img_path, image = self.get_target(data_id)
        boxes = np.float32(boxes)
        image, boxes = random_scale(image, boxes, self.input_size)
        big_image, small_image, characters = self.resize(image, boxes, big_side=self.input_size)  # Resize the image
        character_labels = np.array([0]*len(characters))
        affinities = generate_affinity_box(characters.copy(), words)
        affinity_labels = np.array([1]*len(affinities))

        boxes = np.concatenate((characters, affinities), axis = 0)
        labels = np.concatenate((character_labels, affinity_labels), axis = 0)

        mask = np.zeros((small_image.shape[0], small_image.shape[1], self.num_classes*2), dtype=np.float32)
        area = np.zeros((small_image.shape[0], small_image.shape[1], self.num_classes*2), dtype=np.float32)
        boxes = np.reshape(boxes, (-1, 8))
        
        total_size = 1

        if boxes is not None:
            labels = labels.astype(np.int32)

            numobj = max(len(boxes), 1)
            total_size = self.sum_of_size(boxes)
            
            for box, label in zip(boxes, labels):
                mask, area = tricubemask(mask, area, box, total_size/numobj, label)

        big_image, mask, area = self.transform(big_image, mask, area)

        if self.evaluation: # evaluation mode
            return img, img_path, boxes, labels

        img = torch.from_numpy(big_image.astype(np.float32)).permute(2, 0, 1)
        mask = torch.from_numpy(mask.astype(np.float32))
        area = torch.from_numpy(area.astype(np.float32))
        total_size = torch.from_numpy(np.array([total_size], dtype=np.float32))

        return img, mask, area, total_size
        
        
    def __len__(self):
        return len(self.ids)

    def sum_of_size(self, boxes):
        size_sum = 0
        
        for (x1, y1, x2, y2, x3, y3, x4, y4) in boxes:
            if x1*x2*x3*x4*y1*y2*y3*y4 < 0:
                continue

            mask_w = max(self.distance([x1, y1], [x2, y2]), self.distance([x3, y3], [x4, y4]))
            mask_h = max(self.distance([x3, y3], [x2, y2]), self.distance([x1, y1], [x4, y4]))
            size_sum = size_sum + mask_w*mask_h
            
        return size_sum


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
