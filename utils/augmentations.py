import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, area):
        for t in self.transforms:
            img, mask, area = t(img, mask, area)
        return img, mask, area

class ConvertFromInts(object):
    def __call__(self, image, mask=None, area=None):
        return image.astype(np.float32), mask, area

class Normalize(object):
    def __init__(self, mean=(0.485,0.456,0.406), var=(0.229,0.224,0.225)):
        self.mean = np.array(mean, dtype=np.float32)
        self.var = np.array(var, dtype=np.float32)

    def __call__(self, image, mask, area):
        image = image.astype(np.float32)
        image /= 255.
        image -= self.mean
        image /= self.var
        return image, mask, area



class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, mask, area):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
            image[image>255] = 255
            image[image<0] = 0
        return image, mask, area


class RandomHue(object):
    def __init__(self, delta=36.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, mask=None, area=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, mask, area


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, mask=None, area=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, mask, area


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, mask=None, area=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, mask, area


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, mask=None, area=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
            image[image>255] = 255
            image[image<0] = 0
        return image, mask, area


class RandomBrightness(object):
    def __init__(self, delta=16):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, mask=None, area=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
            image[image>255] = 255
            image[image<0] = 0
        return image, mask, area


class ToCV2Image(object):
    def __call__(self, tensor, mask=None, area=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), mask, area


class RandomMirror(object):
    def __call__(self, image, mask, area):
        _, width, _ = image.shape
        if random.randint(2):
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
            area = np.flip(area, axis=1).copy()
        return image, mask, area


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image

class Rotate(object):
    def __init__(self, mean):
        self.angle_option = [0, -90, 180, 90]
        self.rotate_prob = 3     # 5 -> 0.2

    def __call__(self, image, mask, area):

        if random.randint(self.rotate_prob) == 0:
            angle = random.choice(self.angle_option)
            if angle == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                area = cv2.rotate(area, cv2.ROTATE_90_CLOCKWISE)
            elif angle == -90:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                area = cv2.rotate(area, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif angle == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
                mask = cv2.rotate(mask, cv2.ROTATE_180)
                area = cv2.rotate(area, cv2.ROTATE_180)
            
        return image, mask, area


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            # RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, mask=None, area=None):
        im = image.copy()
        im, mask, area = self.rand_brightness(im, mask, area)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, mask, area = distort(im, mask, area)
        # im, mask, area = self.rand_light_noise(im, mask, area)
        
        im = np.clip(im, 0., 255.)
        return im, mask, area


# class Resize(object):
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, image, mask=None, area=None):
#         target_h, target_w = self.size, self.size

#         image = cv2.resize(image.astype(np.uint8), (target_w, target_h))
#         mask = cv2.resize(mask, (int(target_w//2), int(target_h//2)))
#         area = cv2.resize(area, (int(target_w//2), int(target_h//2)))

#         return image, mask, area


class Augmentation(object):
    def __init__(self, size, mean=(104, 117, 123), var=(0.229,0.224,0.225)):
        self.mean = mean
        self.size = size

        self.augment = Compose([
            ConvertFromInts(),
            Rotate(self.mean),
            PhotometricDistort(),
            RandomMirror(),
            # Resize(self.size), 
            Normalize(mean, var),
        ])

    def __call__(self, img, mask, area):
        if mask is not None:
            mask = np.array(mask)
            
        if area is not None:
            area = np.array(area)
        
        return self.augment(img, mask, area)

    
class Augmentation_test(object):
    def __init__(self, size, mean=(104, 117, 123), var=(0.229,0.224,0.225)):
        self.mean = mean
        self.size = size

        self.augment = Compose([
            ConvertFromInts(),
            # Resize(self.size),
            Normalize(mean, var),
        ])

    def __call__(self, img, mask=None, area=None):
        
        if mask is not None:
            mask = np.array(mask)
            
        if area is not None:
            area = np.array(area)
        
        return self.augment(img, mask, area)