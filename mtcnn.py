import keras
import numpy as np

import cv2
import models
import utils


class MTCNN(object):
    def __init__(self, min_face_size: int=24, scale: float=0.709):
        self.min_face_size = min_face_size
        self.scale_factor = scale
    
    @staticmethod
    def generate_bboxes_with_scores(cls_map, scale, threshold=0.5, size=12, stride=2):
        """
            generate bounding boxes from score map

            @param cls_map: PNet's output feature map for classification
            @param scale: the scale of the image feed to PNet, used to 
                convert bbox coordinates of the resized image into coordinates
                of the original image 
            @param threshold: classification score threshold
            @param size: bbox size (size, size) (default 12)
            @param stride: the stride for bbox generation (default 2)
        """
        assert len(cls_map.shape) == 2

        indices = np.where(cls_map >= threshold)
        bboxes = np.concatenate((
            ((indices[1] * stride) / scale).reshape(-1, 1),
            ((indices[0] * stride) / scale).reshape(-1, 1),
            ((indices[1] * stride + size) / scale).reshape(-1, 1),
            ((indices[0] * stride + size) / scale).reshape(-1, 1),
            cls_map[indices].reshape(-1, 1)
        ), axis=1)
        return bboxes, indices


    def get_image_pyramid_scales(self, min_size: int, img_size: tuple):
        m = min(img_size)
        scales = []
        scale = 1
        
        while m >= min_size:
            scales.append(scale)
            scale *= self.scale_factor
            m *= self.scale_factor
        return scales
    

    @staticmethod
    def non_maximum_suppression(boxes, threshold: float, mode: str='union'):
        """
            Non Maximum Suppression

            @return: indices of remained boxes
        """
        assert boxes.shape[1] == 5
        assert mode in ('union', 'minimum')

        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 2], boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        result = []
        while order.size > 0:
            i = order[0]
            others = order[1:]
            result.append(i)

            ix1 = np.maximum(x1[i], x1[others])
            iy1 = np.maximum(y1[i], y1[others])
            ix2 = np.minimum(x2[i], x2[others])
            iy2 = np.minimum(y2[i], y2[others])

            w = np.maximum(ix2 - ix1 + 1, 0)
            h = np.maximum(iy2 - iy1 + 1, 0)
            intersect = w * h

            if mode == 'union':
                iou = intersect / (areas[i] + areas[others] - intersect)
            elif mode == 'minimum':
                iou = intersect / (np.minimum(areas[i], areas[others]))
            
            i = np.where(iou <= threshold)[0]
            order = order[i + 1]
        return np.array(result, dtype=np.int32)
    
    
    @staticmethod
    def refine_bboxes(boxes, reg_offsets, transform=lambda x: x.astype(np.int32).reshape(-1, 1)):
        w = boxes[:, 2] - boxes[:, 0] + 1
        h = boxes[:, 3] - boxes[:, 1] + 1

        refined_boxes = np.concatenate((
            transform(boxes[:, 0] + reg_offsets[:, 0] * w),
            transform(boxes[:, 1] + reg_offsets[:, 1] * h),
            transform(boxes[:, 2] + reg_offsets[:, 2] * w),
            transform(boxes[:, 3] + reg_offsets[:, 3] * h)
        ), axis=1)
        return refined_boxes


    def stage_PNet(self, model, img):
        h, w, _ = img.shape
        img_size = (w, h)

        boxes_tot = np.empty((0, 5))
        reg_offsets = np.empty((0, 4))

        scales = self.get_image_pyramid_scales(self.min_face_size, img_size)

        print(scales)

        for scale in scales:
            resized = utils.scale_image(img, scale)
            normalized = utils.normalize_image(resized)
            net_input = np.expand_dims(normalized, 0)

            cls_map, reg_map, _ = model.predict(net_input)
            cls_map = cls_map.squeeze()[:, :, 1] # here
            reg_map = reg_map.squeeze()

            boxes, indices = self.generate_bboxes_with_scores(cls_map, scale)
            reg_deltas = reg_map[indices]

            indices = self.non_maximum_suppression(boxes, 0.5, 'union')
            boxes_tot = np.append(boxes_tot, boxes[indices], axis=0)
            reg_offsets = np.append(reg_offsets, reg_deltas[indices], axis=0)
        

        indices = self.non_maximum_suppression(boxes_tot, 0.7, 'union')
        boxes_tot = boxes_tot[indices]
        reg_offsets = reg_offsets[indices]

        # refine bounding boxes
        refined_boxes = self.refine_bboxes(boxes_tot, reg_offsets)
        return refined_boxes


if __name__ == '__main__':

    img = cv2.imread('004.jpg')

    mtcnn = MTCNN()

    PNet = models.PNet(training=False)
    PNet.load_weights('saved_models/PNet-000.h5')

    bboxes = mtcnn.stage_PNet(PNet, img)
    print('bbox count: ', bboxes.shape[0])
    for box in bboxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
    
    print(bboxes)
    cv2.imshow('detection', img)
    cv2.waitKey(0)