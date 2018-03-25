import numpy as np

import cv2


def IoU_single(box1, box2):
    """
        calculate IoU of two bounding boxes
    """
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    w = max(x2 - x1 + 1, 0)
    h = max(y2 - y1 + 1, 0)

    inter = w * h
    union = area1 + area2 - inter
    return inter / union


def IoU(box, boxes):
    """
        calculate IoU of a bounding box over an array of bounding boxes

        @param box: (x1, y1, x2, y2)
        @param boxes: numpy array, shape (n, 4)

        return: an scaler array of shape (n)
    """
    assert len(boxes.shape) == 2

    area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(x2 - x1 + 1, 0)
    h = np.maximum(y2 - y1 + 1, 0)

    inter = w * h
    union = area + areas - inter
    return inter / union


def convert_to_square_bbox(boxes):
    """
        convert rectangle bounding boxes into square bounding boxes
        WARNING: border value is not processed

        @param boxes: numpy array, shape (n, 4)
    """
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    l = np.maximum(w, h)

    res = np.empty_like(boxes)
    res[:, 0] = boxes[:, 0] + w / 2 - l / 2
    res[:, 1] = boxes[:, 1] + h / 2 - l / 2
    res[:, 2] = res[:, 0] + l
    res[:, 3] = res[:, 1] + l
    return res


def crop_image(img, box):
    """
        crop the image according to the bounding box

        @param img: numpy array
        @param box: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box
    return img[y1: y2 + 1, x1: x2 + 1, :]


def convert_bbox(box, kind=True):
    """
        (x1, y1, x2, y2) --> (x1, y1, w, h) (kind=True)
        or
        (x1, y1, w, h) --> (x1, y1, x2, y2) (kind=False)
    """
    a, b, c, d = box
    if kind:
        return (a, b, c - a + 1, d - b + 1)
    else:
        return (a, b, a + c - 1, b + d - 1)


def is_valid_bbox(box, img_size):
    x1, y1, x2, y2 = box
    w, h = img_size
    return x1 >= 0 and y1 >= 0 and x2 < w and y2 < h


def unpack_bbox(box):
    x1, y1, x2, y2 = box
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    assert w > 0 and h > 0
    return x1, y1, x2, y2, w, h


def bbox_size(box):
    x1, y1, x2, y2 = box
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    assert w > 0 and h > 0
    return (w, h)


def bbox_global_negative_sampling(box, img_size, dst_size):
    """
        crop bounding boxes for negative samples

        @param box: bounding box
        @param img_size: original full image size
        @param dst_size: ultimate desired image size (dst_size, dst_size)

        note: no validation check
    """
    w, h = bbox_size(box)
    w_img, h_img = img_size

    sz = np.random.random_integers(min(dst_size, w, h), max(dst_size, w, h))
    if w_img <= sz or h_img <= sz:
        return (-1, -1, -1, -1) # invalid. not elegant
    nx = np.random.randint(0, w_img - sz)
    ny = np.random.randint(0, h_img - sz)

    return (nx, ny, nx + sz, ny + sz)


def bbox_local_negative_sampling(box, dst_size):
    """
        crop bounding boxes for negative samples

        @param box: bounding box (x1, y1, x2, y2)
        @param dst_size: ultimate desired image size (dst_size, dst_size)

        note: no validation check
    """
    x1, y1, w, h = convert_bbox(box, True)
    min_sz = min(dst_size, w, h)
    max_sz = max(dst_size, w, h)
    
    sz = np.random.random_integers(min_sz, max_sz * 2)
    dx = np.random.randint(-max_sz, max_sz)
    dy = np.random.randint(-max_sz, max_sz)

    nx1 = max(x1 + dx, 0)
    ny1 = max(y1 + dy, 0)
    return (nx1, ny1, nx1 + sz, ny1 + sz)


def bbox_positive_sampling(box, size_scale_rate=0.8, delta_scale_rate=0.15):
    """
        crop bounding boxes for positive and partial samples

        @param box: bounding box (x1, y1, x2, y2)
        @param size_scale_rate: float between 0 and 1, controls the range of bbox size
        @param delta_scale_rate: float between 0 and 1, controls bbox coordinate offsets

        note: no validation check
    """
    assert 0 < size_scale_rate <= 1
    assert 0 <= delta_scale_rate < 1
    x1, y1, w, h = convert_bbox(box, True)

    sz = np.random.random_integers(
        np.floor(min(w, h) * size_scale_rate),
        np.ceil(max(w, h) / size_scale_rate)
    )
    dx = np.random.random_integers(-w * delta_scale_rate, w * delta_scale_rate)
    dy = np.random.random_integers(-h * delta_scale_rate, h * delta_scale_rate)

    nx1 = max(x1 + dx + w / 2 - sz / 2, 0)
    ny1 = max(y1 + dy + h / 2 - sz / 2, 0)
    return tuple(map(int, (nx1, ny1, nx1 + sz, ny1 + sz)))


def generate_bbox_from_landmarks(img, ldmks, scale=1.85):
    """
        @param ldmks: might be a numpy array, shape (n, 2)
    """
    h_img, w_img, _ = img.shape
    x, y, w, h = cv2.boundingRect(np.array(ldmks))
    l = max(w, h) * scale

    x1 = x + w / 2 - l / 2
    y1 = y + h / 2 - l / 2
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)

    x2 = x1 + l
    y2 = y1 + l
    x2 = min(int(x2), w_img - 1)
    y2 = min(int(y2), h_img - 1)

    return (x1, y1, x2, y2)


def rotate_facial_landmarks(img, ldmks, bbox, angle):
    """
        rotate facial landmarks and the original image

        @param ldmks: numpy array, shape (n, 2)
        @param bbox: bounding box for face
        @param angle: rotate angle, counter clockwise

        return: rotated image and landmarks

        X = [x, y]^T
        A = [a0, a1; a2, a3] b = [b0, b1]^T

        X' = AX + b = [a0 * x + a1 * y + b0; a2 * x + a3 * y + b1]
    """
    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rot_img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))

    r = rot_mat.ravel()
    transform = lambda x: (x[0] * r[0] + x[1] * r[1] + r[2], x[0] * r[3] + x[1] * r[4] + r[5])
    to_int = lambda x: tuple(map(int, x))
    rot_ldmks = list(map(to_int, map(transform, ldmks)))

    return rot_img, rot_ldmks


def crop_bbox_for_facial_landmarks(ldmks, size_scale_rate=1.85, delta_neg_rate=0.45):
    """
        note: border
    """
    assert size_scale_rate >= 1

    x, y, w, h = cv2.boundingRect(np.array(ldmks))
    l = max(w, h)

    x0 = int(x + w / 2 - l / 2)
    y0 = int(y + h / 2 - l / 2)

    while True:
        dx = np.random.random_integers(-l * delta_neg_rate, 0)
        dy = np.random.random_integers(-l * delta_neg_rate, 0)
        sz = np.random.random_integers(l, l * size_scale_rate)

        x1 = x0 + dx
        y1 = y0 + dy
        x2 = x1 + sz
        y2 = y1 + sz

        if x1 <= x and y1 <= y and x2 >= x + w and y2 >= y + h:
            return (x1, y1, x2, y2)


def adjust_lighting_naive(img, scale=0.65):
    assert 0 < scale < 1
    high = 255 / (img.max() + 1)
    low = high * scale
    alpha = np.random.uniform(low, high)
    img = img * alpha
    return img.astype(np.uint8)


def adjust_hue_and_saturation(img, h=5, s=15):
    assert h >= 0 and s >= 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 0] += np.random.random_integers(-h, h)
    hsv[:, :, 1] += np.random.random_integers(-s, s)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bgr


def normalize_image(img):
    return (img - 127.5) / 128


if __name__ == '__main__':
    #box = np.random.random_integers(0, 100, size=(4,))
    #boxes = np.random.random_integers(0, 100, size=(10, 4))
    #boxes = np.array([(10, 30, 50, 60), (20, 5, 40, 45)])
    #print(boxes)
    #boxes = convert_to_square_bbox(boxes)
    #print(boxes)
    img = cv2.imread('000001.jpg')
    ldmks = np.array([(69, 109), (106, 113), (77, 142), (73, 152), (108, 154)])
    #box = generate_bbox_from_landmarks(img, ldmks)
    #x1, y1, x2, y2 = box
    #for angle in range(-30, 30):
    #    rotate_facial_landmarks(img, ldmks, box, angle)

    '''
    rot_img, rot_ldmks = rotate_facial_landmarks(img, ldmks, box, 30)

    for pt in rot_ldmks:
        cv2.circle(rot_img, pt, 1, (0, 255, 0), 2)

    cropped = crop_image(rot_img, box)

    for i in range(20):
        rd = np.random.rand(3)
        aug = adjust_hue_and_saturation(cropped)
        cv2.imshow('cr', aug)
        cv2.waitKey(0)

    x = map(adjust_hue_and_saturation, [cropped])
    x = tuple(x)
    '''

    for pt in ldmks:
        pt = tuple(pt)
        cv2.circle(img, pt, 1, (0, 255, 0), 2)

    for _ in range(10):
        box = crop_bbox_for_facial_landmarks(ldmks)
        if is_valid_bbox(box, (img.shape[1], img.shape[0])):
            angle = np.random.random_integers(-15, 15)
            aug, r_ldmks = rotate_facial_landmarks(img, ldmks, box, angle)
            aug = crop_image(aug, box)
            aug = adjust_hue_and_saturation(aug)
            aug = adjust_lighting_naive(aug)
            cv2.imshow('aug', aug)
            cv2.waitKey(0)

    pass
