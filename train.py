import argparse
import os
import time

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


def train(model, weight_to_save, logs_to_save, pretrained_weights, **kwargs):
    # training configuration and paths
    save_dir = kwargs['save_dir'] if kwargs.get('save_dir') else 'saved_models'
    log_dir = kwargs['log_dir'] if kwargs.get('log_dir') else 'logs'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    if pretrained_weights:
        if os.path.exists(pretrained_weights):
            model.load_weights(pretrained_weights)
        elif os.path.exists(os.path.join(save_dir, pretrained_weights)):
            model.load_weights(os.path.join(save_dir, pretrained_weights))
    
    weights_path = os.path.join(save_dir, weight_to_save)
    
    # model compilation
    optimizer = kwargs['optimizer']
    loss = kwargs['loss']
    metrics = kwargs['metrics']
    loss_weights = kwargs['loss_weights']

    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)

    # prepare callbacks
    csv_logger = keras.callbacks.CSVLogger(os.path.join(log_dir, logs_to_save))
    callbacks = [csv_logger]
    if kwargs.get('callbacks'):
        callbacks.extend(kwargs['callbacks'])
    
    # configure multiprocessing and epochs
    use_multiprocessing = kwargs['use_multiprocessing'] if kwargs.get('use_multiprocessing') else False
    workers = kwargs['workers'] if kwargs.get('workers') else 1

    epochs = kwargs['epochs'] if kwargs.get('epochs') else 1
    steps_per_epoch = kwargs['steps_per_epoch'] if kwargs.get('steps_per_epoch') else 1000
    initial_epoch = kwargs['initial_epoch'] if kwargs.get('initial_epoch') else 0
    if initial_epoch:
        kwargs['skip'] = initial_epoch * steps_per_epoch

    # data generator
    datagen = data_loader.augmented_data_generator(**kwargs)

    # training
    start_time = time.time()
    try:
        model.fit_generator(datagen, steps_per_epoch=steps_per_epoch, epochs=epochs,
            use_multiprocessing=use_multiprocessing, workers=workers,
            callbacks=callbacks, shuffle=True)
    except KeyboardInterrupt:
        print('Keyboard Interrupt received. stop training.')
    finally:
        # save weights
        model.save_weights(weights_path)
        print('model weights saved at %s.' %(weights_path))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('training finished. time: %d h %d m %g s'
            %(elapsed_time // 3600, (elapsed_time % 3600) // 60, elapsed_time % 60))


if __name__ == '__main__':
    PNet = models.P_Net_alter1()
    sgd = keras.optimizers.SGD(lr=1e-4, momentum=0.9)
    loss_PNet = {'face_cls': face_cls_loss, 'bbox_reg': bbox_reg_mse_loss, 'ldmk_reg': ldmk_reg_mse_loss}
    metrics_PNet = {'face_cls': [recall, accuracy]}
    loss_weights_PNet = {'face_cls': 1.0, 'bbox_reg': 0.5, 'ldmk_reg': 0.5}

    train(PNet, 'PNet-alter1-999.h5', 'PNet-alter1-train-999.csv', 'PNet-alter1-001.h5',
        optimizer=sgd, loss=loss_PNet, loss_weights=loss_weights_PNet, metrics=metrics_PNet,
        initial_epoch=100, epochs=1, use_multiprocessing=True, workers=4, dst_size=12)
    