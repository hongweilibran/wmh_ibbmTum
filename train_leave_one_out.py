#Codes for validating the WMH Segmetation Challenge public training Datasets. The algorithm won the MICCAI WMH Segmentation Challenge 2017.
#Codes are written by Mr. Hongwei Li (hongwei.li@tum.de) and Mr. Gongfa Jiang (jianggfa@mail2.sysu.edu.cn). They are PhD students in Technical University of Munich and Sun Yat-sen University.
#Please cite our paper titled 'Fully Convolutional Networks Ensembles for White Matter Hyperintensities Segmentation in MR Images' if you found it is useful to your research.
#Please contact me if there is any bug you want to report or any details you would like to know. 

import os
import time
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, BatchNormalization, Activation
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import apply_transform, transform_matrix_offset_center
import warnings
K.set_image_data_format('channels_last')

smooth=1.

def dice_coef_for_training(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef_for_training(y_true, y_pred)

def conv_bn_relu(nd, k=3, inputs=None):
    conv = Conv2D(nd, k, padding='same')(inputs) #, kernel_initializer='he_normal'
    #bn = BatchNormalization()(conv)
    relu = Activation('relu')(conv)
    return relu

def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)
#define U-Net architecture
def get_unet(img_shape = None, first5=True):
        inputs = Input(shape = img_shape)
        concat_axis = -1

        if first5: filters = 5
        else: filters = 3
        conv1 = conv_bn_relu(64, filters, inputs)
        conv1 = conv_bn_relu(64, filters, conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = conv_bn_relu(96, 3, pool1)
        conv2 = conv_bn_relu(96, 3, conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = conv_bn_relu(128, 3, pool2)
        conv3 = conv_bn_relu(128, 3, conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = conv_bn_relu(256, 3, pool3)
        conv4 = conv_bn_relu(256, 4, conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = conv_bn_relu(512, 3, pool4)
        conv5 = conv_bn_relu(512, 3, conv5)

        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        ch, cw = get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = conv_bn_relu(256, 3, up6)
        conv6 = conv_bn_relu(256, 3, conv6)

        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        ch, cw = get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = conv_bn_relu(128, 3, up7)
        conv7 = conv_bn_relu(128, 3, conv7)

        up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        ch, cw = get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = conv_bn_relu(96, 3, up8)
        conv8 = conv_bn_relu(96, 3, conv8)

        up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        ch, cw = get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = conv_bn_relu(64, 3, up9)
        conv9 = conv_bn_relu(64, 3, conv9)

        ch, cw = get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=(ch, cw))(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9) #, kernel_initializer='he_normal'
        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)

        return model

def augmentation(x_0, x_1, y):
    theta = (np.random.uniform(-15, 15) * np.pi) / 180.
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    shear = np.random.uniform(-.1, .1)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    zx, zy = np.random.uniform(.9, 1.1, 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    augmentation_matrix = np.dot(np.dot(rotation_matrix, shear_matrix), zoom_matrix)
    transform_matrix = transform_matrix_offset_center(augmentation_matrix, x_0.shape[0], x_0.shape[1])
    x_0 = apply_transform(x_0[..., np.newaxis], transform_matrix, channel_axis=2)
    x_1 = apply_transform(x_1[..., np.newaxis], transform_matrix, channel_axis=2)
    y = apply_transform(y[..., np.newaxis], transform_matrix, channel_axis=2)
    return x_0[..., 0], x_1[..., 0], y[..., 0]

#train single model on the training set
def train_leave_one_out(images, masks, patient=0, flair=True, t1=True, full=True, first5=True, aug=True, verbose=False):
    if full:
        if patient < 40:
            images = np.delete(images, range(patient*38, (patient+1)*38), axis=0)
            masks = np.delete(masks, range(patient*38, (patient+1)*38), axis=0)
        else:
            images = np.delete(images, range(1520+(patient-40)*63, 1520+(patient-39)*63), axis=0)
            masks = np.delete(masks, range(1520+(patient-40)*63, 1520+(patient-39)*63), axis=0)
    else:
        if patient < 20:
            images = images[:760, ...]
            masks = masks[:760, ...]
            images = np.delete(images, range(patient*38, (patient+1)*38), axis=0)
            masks = np.delete(masks, range(patient*38, (patient+1)*38), axis=0)
        elif patient < 40:
            images = images[760:1520, ...]
            masks = masks[760:1520, ...]
            images = np.delete(images, range((patient-20)*38, (patient-19)*38), axis=0)
            masks = np.delete(masks, range((patient-20)*38, (patient-19)*38), axis=0)
        else:
            images = images[1520:, ...]
            masks = masks[1520:, ...]
            images = np.delete(images, range((patient-40)*63, (patient-39)*63), axis=0)
            masks = np.delete(masks, range((patient-40)*63, (patient-39)*63), axis=0)
    if aug:
        images = np.concatenate((images, images[..., ::-1, :]), axis=0)
        masks = np.concatenate((masks, masks[..., ::-1, :]), axis=0)
    samples_num = images.shape[0]
    row = images.shape[1]
    col = images.shape[2]
    if aug: epoch = 50
    else: epoch = 200
    batch_size = 30

    img_shape = (row, col, flair+t1)
    model = get_unet(img_shape, first5)
    current_epoch = 1
    while current_epoch <= epoch:
        print 'Epoch ', str(current_epoch), '/', str(epoch)
        if aug:
            images_aug = np.zeros(images.shape, dtype=np.float32)
            masks_aug = np.zeros(masks.shape, dtype=np.float32)
            for i in range(samples_num):
                images_aug[i, ..., 0], images_aug[i, ..., 1], masks_aug[i, ..., 0] = augmentation(images[i, ..., 0], images[i, ..., 1], masks[i, ..., 0])
            image = np.concatenate((images, images_aug), axis=0)
            mask = np.concatenate((masks, masks_aug), axis=0)
        else:
            image = images.copy()
            mask = masks.copy()
        if not flair: image = image[..., 1:2].copy()
        if not t1: image = image[..., 0:1].copy()
        history = model.fit(image, mask, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        current_epoch += 1
        if history.history['loss'][-1] > 0.99:
            model = get_unet(img_shape, first5)
            current_epoch = 1
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if full: model_path += 'Full_'
    else:
        if patient < 20: model_path += 'Utrecht_'
        elif patient < 40: model_path += 'Singapore_'
        else: model_path += 'GE3T_'
    if flair: model_path += 'Flair_'
    if t1: model_path += 'T1_'
    if first5: model_path += '5_'
    else: model_path += '3_'
    if aug: model_path += 'Augmentation/'
    else: model_path += 'No_Augmentation/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path += str(patient) + '.h5'
    model.save_weights(model_path)
    print 'Model saved to ', model_path

#leave-one-out evaluation
def main():
    warnings.filterwarnings("ignore")
    images = np.load('images_three_datasets_sorted.npy')
    masks = np.load('masks_three_datasets_sorted.npy')
    patient_num  = 60
    for patient in range(0, patient_num):
        train_leave_one_out(images, masks, patient=patient, full=True, verbose=True)

if __name__=='__main__':
    main()
