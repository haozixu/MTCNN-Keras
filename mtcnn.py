import keras
import numpy as np

import cv2
import models
import utils


def apply_softmax(score_map):
    assert len(score_map.shape) == 3 and score_map.shape[-1] == 3

    score_map = score_map[:, :, 1:]
    score_exp = np.exp(score_map)
    exp_sum = np.sum(score_exp, axis=-1, keepdims=True)
    exp_sum = np.repeat(exp_sum, 2, axis=-1)
    return score_exp / exp_sum


if __name__ == '__main__':
    ## test

    img = cv2.imread('004.jpg')

    P_Net = models.P_Net_alter1(training=False)
    P_Net.load_weights('saved_models/PNet-alter1-000.h5')

    net_input = np.expand_dims(utils.normalize_image(img), 0)
    score_map, _, _ = P_Net.predict(net_input)

    score_map = score_map.squeeze()
    #score_map = apply_softmax(score_map)

    print(score_map, score_map.shape)

    cv2.imshow('score_map', score_map[:, :, 1])
    cv2.waitKey(0)
