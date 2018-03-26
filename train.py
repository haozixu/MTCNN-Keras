import argparse
import os

import keras
import tensorflow as tf

import data_loader
import models
from data_loader import SampleType


def smooth_l1_loss(y_true, y_pred, sigma=1.0):
    sigma2 = sigma ** 2
    thresold = 1 / sigma

    abs_error = tf.abs(y_true - y_pred)
    loss_smaller = 0.5 * sigma2 * tf.square(abs_error)
    loss_greater = abs_error - 0.5 / sigma2
    loss = tf.where(abs_error < thresold, loss_smaller, loss_greater)
    return tf.reduce_mean(loss)


def face_cls_filter(y_true, y_pred):
    label = y_true[:, 0]
    mask = tf.logical_or(
        tf.equal(label, SampleType.positive.value),
        tf.equal(label, SampleType.negative.value)
    )
    cls_true = tf.boolean_mask(y_true, mask)[:, 1:]
    cls_pred = tf.boolean_mask(y_pred, mask)[:, 1:]
    return cls_true, cls_pred


def bbox_reg_filter(y_true, y_pred):
    label = y_true[:, 0]
    mask = tf.logical_or(
        tf.equal(label, SampleType.positive.value),
        tf.equal(label, SampleType.partial.value)
    )
    bbox_reg_true = tf.boolean_mask(y_true, mask)[:, 1:]
    bbox_reg_pred = tf.boolean_mask(y_pred, mask)[:, 1:]
    return bbox_reg_true, bbox_reg_pred


def ldmk_reg_filter(y_true, y_pred):
    label = y_true[:, 0]
    mask = tf.equal(label, SampleType.landmark.value)
    ldmk_reg_true = tf.boolean_mask(y_true, mask)[:, 1:]
    ldmk_reg_pred = tf.boolean_mask(y_pred, mask)[:, 1:]
    return ldmk_reg_true, ldmk_reg_pred


def face_cls_loss(y_true, y_pred):
    cls_true, cls_pred = face_cls_filter(y_true, y_pred)
    cls_loss = tf.nn.softmax_cross_entropy_with_logits(logits=cls_pred, labels=cls_true)
    loss = tf.reduce_mean(cls_loss)
    return loss


def bbox_reg_mse_loss(y_true, y_pred):
    bbox_reg_true, bbox_reg_pred = bbox_reg_filter(y_true, y_pred)
    bbox_reg_loss = tf.losses.mean_squared_error(bbox_reg_true, bbox_reg_pred)
    return bbox_reg_loss


def bbox_reg_smooth_l1_loss(y_true, y_pred):
    bbox_reg_true, bbox_reg_pred = bbox_reg_filter(y_true, y_pred)
    bbox_reg_loss = smooth_l1_loss(bbox_reg_true, bbox_reg_pred)
    return bbox_reg_loss


def ldmk_reg_mse_loss(y_true, y_pred):
    ldmk_reg_true, ldmk_reg_pred = ldmk_reg_filter(y_true, y_pred)
    ldmk_reg_loss = tf.losses.mean_squared_error(ldmk_reg_true, ldmk_reg_pred)
    return ldmk_reg_loss


def ldmk_reg_smooth_l1_loss(y_true, y_pred):
    ldmk_reg_true, ldmk_reg_pred = ldmk_reg_filter(y_true, y_pred)
    ldmk_reg_loss = smooth_l1_loss(ldmk_reg_true, ldmk_reg_pred)
    return ldmk_reg_loss


def accuracy(y_true, y_pred):
    cls_true, cls_pred = face_cls_filter(y_true, y_pred)
    cls_true = tf.argmax(cls_true, axis=-1)
    cls_pred = tf.argmax(cls_pred, axis=-1)
    right_predictions = tf.to_int32(tf.equal(cls_true, cls_pred))
    accuracy = tf.reduce_sum(right_predictions) / tf.size(right_predictions)
    return accuracy


def recall(y_true, y_pred):
    cls_true, cls_pred = face_cls_filter(y_true, y_pred)
    cls_true = tf.argmax(cls_true, axis=-1)
    cls_pred = tf.argmax(cls_pred, axis=-1)
    true_positives = tf.reduce_sum(cls_true * cls_pred)
    possible_positives = tf.reduce_sum(cls_true)
    recall = true_positives / possible_positives
    return recall


def precision(y_true, y_pred):
    cls_true, cls_pred = face_cls_filter(y_true, y_pred)
    cls_true = tf.argmax(cls_true, axis=-1)
    cls_pred = tf.argmax(cls_pred, axis=-1)
    true_positives = tf.reduce_sum(cls_true * cls_pred)
    predicted_positives = tf.reduce_sum(cls_pred)
    precision = true_positives / predicted_positives
    return precision


def f1_score(y_true, y_pred):
    cls_true, cls_pred = face_cls_filter(y_true, y_pred)
    cls_true = tf.argmax(cls_true, axis=-1)
    cls_pred = tf.argmax(cls_pred, axis=-1)

    true_positives = tf.reduce_sum(cls_true * cls_pred)
    possible_positives = tf.reduce_sum(cls_true)
    predicted_positives = tf.reduce_sum(cls_pred)

    recall = true_positives / possible_positives
    precision = true_positives / predicted_positives
    return 2 * precision * recall / (precision + recall)


if __name__ == '__main__':
    save_dir = 'saved_models'
    log_dir = 'logs'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    model = models.P_Net_alter1()
    
    loss_dict = {'face_cls': face_cls_loss, 'bbox_reg': bbox_reg_mse_loss, 'ldmk_reg': ldmk_reg_mse_loss}
    metrics_dict = {'face_cls': [recall, accuracy]}

    loss_weights = {'face_cls': 1.0, 'bbox_reg': 0.5, 'ldmk_reg': 0.5}

    sgd = keras.optimizers.SGD(lr=1e-3, momentum=0.9)
    adam = keras.optimizers.Adam()
    model.compile(optimizer=sgd, loss=loss_dict, loss_weights=loss_weights, metrics=metrics_dict)

#    model.summary()

    weights_file = 'PNet-alter1-000.h5'
    logs_file = 'PNet-alter1-train.csv'
    csv_logger = keras.callbacks.CSVLogger(os.path.join(log_dir, logs_file))
    datagen = data_loader.augmented_data_generator(dst_size=12, double_aug=False)

    weights_path = os.path.join(save_dir, weights_file)
    if os.path.exists(weights_path):
        model.load_weights(weights_path)

    try:
        model.fit_generator(datagen, steps_per_epoch=1000, epochs=100, 
            workers=4, use_multiprocessing=True, shuffle=True, callbacks=[csv_logger])
    except KeyboardInterrupt:
        print('ctrl-c received!')
    finally:
        model.save_weights(weights_path)
