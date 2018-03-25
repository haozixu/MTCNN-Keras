import keras
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D, Reshape
from keras.layers.advanced_activations import PReLU
from keras.models import Model


def P_Net(training=True):
    inp = Input((12, 12, 3))
    
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(inp)
    x = PReLU(shared_axes=(1, 2), name='prelu1')(x)
    x = MaxPool2D((2, 2), strides=2)(x)
    
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=(1, 2), name='prelu2')(x)
    
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=(1, 2), name='prelu3')(x)
    
    if training:
        face_cls = Conv2D(3, (1, 1))(x)
    else:
        face_cls = Conv2D(2, (1, 1), activation='softmax')(x)
    face_cls = Reshape((3,), name='face_cls')(face_cls)
    bbox_reg = Conv2D(5, (1, 1))(x)
    bbox_reg = Reshape((5,), name='bbox_reg')(bbox_reg)
    ldmk_reg = Conv2D(11, (1, 1))(x)
    ldmk_reg = Reshape((11,), name='ldmk_reg')(ldmk_reg)
    
    return Model(inp, [face_cls, bbox_reg, ldmk_reg])
    

def R_Net(training=True):
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
    
    if training:
        face_cls = Dense(3, name='face_cls')(x)
    else:
        face_cls = Dense(3, activation='softmax', name='face_cls')(x)
    bbox_reg = Dense(5, name='bbox_reg')(x)
    ldmk_reg = Dense(11, name='ldmk_reg')(x)
    
    return Model(inp, [face_cls, bbox_reg, ldmk_reg])

def O_Net(training=True):
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
    
    if training:
        face_cls = Dense(3, name='face_cls')(x)
    else:
        face_cls = Dense(3, activation='softmax', name='face_cls')(x)
    bbox_reg = Dense(5, name='bbox_reg')(x)
    ldmk_reg = Dense(11, name='ldmk_reg')(x)
    
    return Model(inp, [face_cls, bbox_reg, ldmk_reg])
    

if __name__ == '__main__':
    model = O_Net()
    model.summary()
