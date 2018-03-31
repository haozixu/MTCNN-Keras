import keras
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D, Reshape, Lambda, Activation
from keras.layers.advanced_activations import PReLU
from keras.models import Model
from keras import backend as K


SampleLabelFilter = Lambda(lambda x: x[..., 1:])


def PNet_conv(x, channels):
    k1, k2, k3 = channels

    x = Conv2D(k1, (3, 3), strides=1, padding='valid', name='conv1')(x)
    x = PReLU(shared_axes=(1, 2), name='prelu1')(x)
    x = MaxPool2D((2, 2), strides=2)(x)
    
    x = Conv2D(k2, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=(1, 2), name='prelu2')(x)
    
    x = Conv2D(k3, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=(1, 2), name='prelu3')(x)

    return x


def PNet_output(x, training):
    if training:
        face_cls = Conv2D(3, (1, 1), name='face_cls_conv')(x)
        bbox_reg = Conv2D(5, (1, 1), name='bbox_reg_conv')(x)
        ldmk_reg = Conv2D(11, (1, 1), name='ldmk_reg_conv')(x)
        
        face_cls = Reshape((3,), name='face_cls')(face_cls)
        bbox_reg = Reshape((5,), name='bbox_reg')(bbox_reg)
        ldmk_reg = Reshape((11,), name='ldmk_reg')(ldmk_reg)
    else:
        face_cls = Conv2D(3, (1, 1), name='face_cls_conv')(x)
        bbox_reg = Conv2D(5, (1, 1), name='bbox_reg_conv')(x)
        ldmk_reg = Conv2D(11, (1, 1), name='ldmk_reg_conv')(x)

        face_cls = SampleLabelFilter(face_cls)
        face_cls = Activation('softmax')(face_cls)
        bbox_reg = SampleLabelFilter(bbox_reg)
        ldmk_reg = SampleLabelFilter(ldmk_reg)
    return face_cls, bbox_reg, ldmk_reg


def RONet_output(x, training):
    face_cls = Dense(3, name='face_cls')(x)
    bbox_reg = Dense(5, name='bbox_reg')(x)
    ldmk_reg = Dense(11, name='ldmk_reg')(x)
    if not training:
        face_cls = SampleLabelFilter(face_cls)
        face_cls = Activation('softmax')(face_cls)
        bbox_reg = SampleLabelFilter(bbox_reg)
        ldmk_reg = SampleLabelFilter(ldmk_reg)
    return face_cls, bbox_reg, ldmk_reg


def PNet(training=True):
    inp = Input((None, None, 3))
    
    x = PNet_conv(inp, (10, 16, 32))
    face_cls, bbox_reg, ldmk_reg = PNet_output(x, training)
    
    return Model(inp, [face_cls, bbox_reg, ldmk_reg])
    

def RNet(training=True):
    inp = Input((24, 24, 3))
    
    x = Conv2D(28, (3, 3), strides=1, padding='same', name='conv1')(inp)
    x = PReLU(shared_axes=(1, 2), name='prelu1')(x)
    x = MaxPool2D((3, 3), strides=2)(x)
    
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=(1, 2), name='prelu2')(x)
    x = MaxPool2D((3, 3), strides=2)(x)
    
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=(1, 2), name='prelu3')(x)
    
    x = Flatten()(x)
    x = Dense(128)(x)
    x = PReLU(name='prelu4')(x)
    
    face_cls, bbox_reg, ldmk_reg = RONet_output(x, training)
    
    return Model(inp, [face_cls, bbox_reg, ldmk_reg])

def ONet(training=True):
    inp = Input((48, 48, 3))
    
    x = Conv2D(32, (3, 3), strides=1, padding='same', name='conv1')(inp)
    x = PReLU(shared_axes=(1, 2), name='prelu1')(x)
    x = MaxPool2D((3, 3), strides=2)(x)
    
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=(1, 2), name='prelu2')(x)
    x = MaxPool2D((3, 3), strides=2)(x)
    
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=(1, 2), name='prelu3')(x)
    x = MaxPool2D((2, 2), strides=2)(x)
    
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=(1, 2), name='prelu4')(x)
    
    x = Flatten()(x)
    x = Dense(256)(x)
    x = PReLU(name='prelu5')(x)
    
    face_cls, bbox_reg, ldmk_reg = RONet_output(x, training)
    
    return Model(inp, [face_cls, bbox_reg, ldmk_reg])
    

def PNet_alter1(training=True):
    inp = Input((None, None, 3))
    x = PNet_conv(inp, (16, 32, 64))
    face_cls, bbox_reg, ldmk_reg = PNet_output(x, training)
    return Model(inp, [face_cls, bbox_reg, ldmk_reg])


def PNet_alter2(training=True):
    inp = Input((None, None, 3))
    x = PNet_conv(inp, (24, 36, 48))
    face_cls, bbox_reg, ldmk_reg = PNet_output(x, training)
    return Model(inp, [face_cls, bbox_reg, ldmk_reg])


if __name__ == '__main__':
    model = PNet_alter1()
    model.summary()
