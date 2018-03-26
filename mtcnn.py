import keras

import models
import utils


if __name__ == '__main__':
    ## test

    import cv2
    import numpy as np

    img = cv2.imread('003.jpg')

    P_Net = models.P_Net_alter1(training=False)

    net_input = np.expand_dims(utils.normalize_image(img), 0)
    score_map, _, _ = P_Net.predict(net_input)
    print(score_map, score_map.shape)

    score_map = score_map.squeeze()[:, :, 2]

    cv2.imshow('score_map', score_map)
    cv2.waitKey(0)

    smap = np.where(score_map > 0.7)
    smap = smap.astype(np.float)
    cv2.imshow('smap', smap)
    cv2.waitKey(0)
