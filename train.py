import argparse
import os
import time

import keras
import tensorflow as tf
from keras import backend as K

import data_loader
import models
from data_loader import SampleType

# NOTE: maybe these functions should be moved to some file called losses.py or metrics.py

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


def face_cls_metric_activation(y_true, y_pred):
    y_true, y_pred = face_cls_filter(y_true, y_pred)
    y_pred = tf.nn.softmax(y_pred)
    cls_true = tf.argmax(y_true, axis=-1)
    cls_pred = tf.argmax(y_pred, axis=-1)
    return cls_true, cls_pred


def accuracy(y_true, y_pred):
    cls_true, cls_pred = face_cls_metric_activation(y_true, y_pred)
    right_predictions = tf.to_int32(tf.equal(cls_true, cls_pred))
    accuracy = tf.reduce_sum(right_predictions) / tf.size(right_predictions)
    return accuracy


def recall(y_true, y_pred):
    cls_true, cls_pred = face_cls_metric_activation(y_true, y_pred)
    true_positives = tf.to_float(tf.reduce_sum(cls_true * cls_pred))
    possible_positives = tf.to_float(tf.reduce_sum(cls_true))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    cls_true, cls_pred = face_cls_metric_activation(y_true, y_pred)
    true_positives = tf.to_float(tf.reduce_sum(cls_true * cls_pred))
    predicted_positives = tf.to_float(tf.reduce_sum(cls_pred))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    cls_true, cls_pred = face_cls_metric_activation(y_true, y_pred)

    true_positives = tf.to_float(tf.reduce_sum(cls_true * cls_pred))
    possible_positives = tf.to_float(tf.reduce_sum(cls_true))
    predicted_positives = tf.to_float(tf.reduce_sum(cls_pred))

    recall = true_positives / (possible_positives + K.epsilon())
    precision = true_positives / (predicted_positives + K.epsilon())
    return 2 * precision * recall / (precision + recall + K.epsilon())


def train(model, weight_to_save, logs_to_save, pretrained_weights, **kwargs):
    # training configuration and paths
    save_dir = kwargs.get('save_dir', 'saved_models')
    logs_dir = kwargs.get('logs_dir', 'logs')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    
    if pretrained_weights:
        if os.path.exists(pretrained_weights):
            model.load_weights(pretrained_weights)
        elif os.path.exists(os.path.join(save_dir, pretrained_weights)):
            model.load_weights(os.path.join(save_dir, pretrained_weights))
    
    weights_path = os.path.join(save_dir, weight_to_save)
    
    # model compilation
    # if these parameters not given, raise KeyErroe
    optimizer = kwargs['optimizer']
    loss = kwargs['loss']
    metrics = kwargs['metrics']
    loss_weights = kwargs['loss_weights']

    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)

    # prepare callbacks
    csv_logger = keras.callbacks.CSVLogger(os.path.join(logs_dir, logs_to_save))
    callbacks = [csv_logger]
    callbacks.extend(kwargs.get('callbacks', []))
    
    # configure multiprocessing and epochs
    use_multiprocessing = kwargs.get('use_multiprocessing', False)
    workers = kwargs.get('workers', 1)

    epochs = kwargs.get('epochs', 1)
    steps_per_epoch = kwargs.get('steps_per_epoch', 1000)
    initial_epoch = kwargs.get('initial_epoch', 0)
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
        print('\nKeyboard Interrupt received. stop training.')
    finally:
        # save weights
        model.save_weights(weights_path)
        print('model weights saved at %s.' %(weights_path))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('training finished. time: %d h %d m %g s'
            %(elapsed_time // 3600, (elapsed_time % 3600) // 60, elapsed_time % 60))


def parse_args():
    parser = argparse.ArgumentParser(description='train PNet, RNet or ONet')

    parser.add_argument('net', choices=('p', 'r', 'o'), help='which network to train')
    parser.add_argument('weights', help='weights to save')
    parser.add_argument('logs', help='logs to save')
    parser.add_argument('--save-dir', help='model weights saving directory')
    parser.add_argument('--logs-dir', help='training logs saving directory')
    parser.add_argument('--pretrained', help='pretrained weights')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--decay', type=float, default=0.0, help='learning rate decay')
    parser.add_argument('--steps', type=int, default=1000, help='steps per epoch')
    parser.add_argument('--epochs', type=int, default=5, help='epochs to train')
    parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch')
    parser.add_argument('--batch-size', help='IGNORED. please modify arguments passed to function train() in train.py manually')
    parser.add_argument('--losses', help='IGNORED. please modify corresponding arguments in train.py manually')
    parser.add_argument('--metrics', help='IGNORED. please modify corresponding arguments in train.py manually')

    args = parser.parse_args()
    return args


def main(args):
    sgd = keras.optimizers.SGD(lr=args.lr, momentum=args.momentum, decay=args.decay)

    if args.net == 'p':
        model = models.PNet()
        net_input_size = 12
        loss = {'face_cls': face_cls_loss, 'bbox_reg': bbox_reg_mse_loss, 'ldmk_reg': ldmk_reg_mse_loss}
        loss_weights = {'face_cls': 1.0, 'bbox_reg': 0.5, 'ldmk_reg': 0.5}
        metrics = {'face_cls': [recall, accuracy]}

    elif args.net == 'r':
        model = models.RNet()
        net_input_size = 24
        loss = {'face_cls': face_cls_loss, 'bbox_reg': bbox_reg_smooth_l1_loss, 'ldmk_reg': ldmk_reg_smooth_l1_loss}
        loss_weights = {'face_cls': 1.0, 'bbox_reg': 0.5, 'ldmk_reg': 0.5}
        metrics = {'face_cls': [precision, f1_score, accuracy]}

    elif args.net == 'o':
        model = models.ONet()
        net_input_size = 48
        loss = {'face_cls': face_cls_loss, 'bbox_reg': bbox_reg_smooth_l1_loss, 'ldmk_reg': ldmk_reg_smooth_l1_loss}
        loss_weights = {'face_cls': 1.0, 'bbox_reg': 0.5, 'ldmk_reg': 1.0}
        metrics = {'face_cls': [precision, f1_score, accuracy]}


    train(model, args.weights, args.logs, args.pretrained, save_dir=args.save_dir, logs_dir=args.log_dir,
        optimizer=sgd, loss=loss, loss_weights=loss_weights, metrics=metrics,
        initial_epoch=args.initial_epoch, epochs=args.epochs,
        use_multiprocessing=True, workers=4, dst_size=net_input_size)


if __name__ == '__main__':
    args = parse_args()
    main(args)

    '''
    PNet = models.PNet_alter2()
    sgd = keras.optimizers.SGD(lr=1e-4, momentum=0.9, decay=0.04)
    loss_PNet = {'face_cls': face_cls_loss, 'bbox_reg': bbox_reg_mse_loss, 'ldmk_reg': ldmk_reg_mse_loss}
    metrics_PNet = {'face_cls': [recall, accuracy]}
    loss_weights_PNet = {'face_cls': 1.0, 'bbox_reg': 0.5, 'ldmk_reg': 0.5}

    train(PNet, 'PNet-alter2-001.h5', 'PNet-alter2-train-001.csv', 'PNet-alter2-000.h5',
        optimizer=sgd, loss=loss_PNet, loss_weights=loss_weights_PNet, metrics=metrics_PNet,
        initial_epoch=100, epochs=100, use_multiprocessing=True, workers=4, dst_size=12)
    '''
