
import cv2
import numpy as np

def line_kernel(size, factor=7):
    s = (size-1)/2
    m, n = [(ss - 1.) / 2. for ss in [size, size]]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    
    h_x = (1 - (abs((x/s)**18)))**factor  # horizontal linecube
    h_y = (1 - (abs((y/s)**3)))**factor  # vertical linecube
    
    heat = h_y.dot(h_x)
    
    heat /= heat.max()
    
    return heat

def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

diameter = 400
gaussian_map = line_kernel(diameter, 7)
# gaussian_poly = np.float32([[0, 0], [0, diameter], [diameter, diameter], [diameter, 0]])
_, binary = cv2.threshold(gaussian_map, 0.1, 1, 0)
np_contours = np.roll(np.array(np.where(binary != 0)), 1, axis=0).transpose().reshape(-1, 2)
x, y, w, h = cv2.boundingRect(np_contours)
gaussian_poly = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


def tricubemask(mask, area, box, size, label):
    if type(size) is tuple:
        size = size[0] * size[1]
        
    H, W = mask.shape[:2]
        
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    
    if x1*x2*x3*x4*y1*y2*y3*y4 < 0:
        return mask, area

    mask_w = max(distance([x1, y1], [x2, y2]), distance([x3, y3], [x4, y4]))
    mask_h = max(distance([x3, y3], [x2, y2]), distance([x1, y1], [x4, y4]))
    
    if mask_w > 0 and mask_h > 0:
        weight_mask = np.zeros((H, W), dtype=np.float32)

        mask_area = max(1, mask_w * mask_h)
        img_area = size

        M = cv2.getPerspectiveTransform(gaussian_poly, box.reshape((4, 2)))
        dst = cv2.warpPerspective(gaussian_map, M, (H, W), flags=cv2.INTER_LINEAR)

        mask_area = (img_area/mask_area)

        weight_mask = cv2.fillPoly(weight_mask, box.astype(np.int32).reshape((-1,4,2)), color=mask_area)

        mask[:, :, label] = np.maximum(mask[:, :, label], dst)
        area[:, :, label] = np.maximum(area[:, :, label], weight_mask)
        
    return mask, area

def craftmask(mask, box):

    H, W = mask.shape[:2]
        
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    
    for i in range(len(box)):
        if box[i] < 0 :
            box[i] = 0


    mask_w = max(distance([x1, y1], [x2, y2]), distance([x3, y3], [x4, y4]))
    mask_h = max(distance([x3, y3], [x2, y2]), distance([x1, y1], [x4, y4]))
    
    if mask_w > 0 and mask_h > 0:

        box = np.array(box, dtype=np.float32)
        M = cv2.getPerspectiveTransform(gaussian_poly, box.reshape((4, 2)))
        dst = cv2.warpPerspective(gaussian_map, M, (H, W), flags=cv2.INTER_LINEAR)

        mask[:, :] = np.maximum(mask[:, :], dst)
        
    return mask