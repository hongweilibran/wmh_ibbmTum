from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
import scipy.ndimage
import scipy.spatial
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# -define u-net architecture--------------------
smooth = 1.

def dice_coef_for_training(y_true, y_pred):
    print(np.shape(y_pred))
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    print(np.shape(y_pred))
    print(np.shape(y_true))
    return -dice_coef_for_training(y_true, y_pred)

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.shape[2] - refer.shape[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.shape[1] - refer.shape[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

def get_unet(img_shape=None):
    
    inputs = Input(shape=img_shape)
    concat_axis = -1
        
    conv1 = Conv2D(64, 5, activation='relu', padding='same', name='conv1_1')(inputs)
    conv1 = Conv2D(64, 5, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(96, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(96, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 4, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)

    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch, cw))(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch, cw))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)

    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch, cw))(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = Conv2D(96, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(96, 3, activation='relu', padding='same')(conv8)

    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=(ch, cw))(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(learning_rate=(1e-4)*2), loss=dice_coef_loss, metrics=[dice_coef_for_training])

    return model

#--------------------------------------------------------------------------------------
def preprocessing(FLAIR_array, T1_array):
    
    brain_mask = np.ndarray(np.shape(FLAIR_array), dtype=np.float32)
    brain_mask[FLAIR_array >= thresh] = 1
    brain_mask[FLAIR_array < thresh] = 0
    for iii in range(np.shape(FLAIR_array)[0]):
        brain_mask[iii, :, :] = scipy.ndimage.morphology.binary_fill_holes(brain_mask[iii, :, :])
    
    # Normalize BOTH arrays BEFORE resizing
    FLAIR_array -= np.mean(FLAIR_array[brain_mask == 1])
    FLAIR_array /= np.std(FLAIR_array[brain_mask == 1])
    
    if two_modalities:
        T1_array -= np.mean(T1_array[brain_mask == 1])
        T1_array /= np.std(T1_array[brain_mask == 1])
    
    rows_o = np.shape(FLAIR_array)[1]
    cols_o = np.shape(FLAIR_array)[2]
    
    # Handle each dimension independently
    row_diff = rows_standard - rows_o
    col_diff = cols_standard - cols_o
    
    # Initialize tracking variables
    row_pad = None
    col_pad = None
    row_crop = None
    col_crop = None
    
    # Handle rows
    if row_diff > 0:  # Need row padding
        pad_row_before = row_diff // 2
        pad_row_after = row_diff - pad_row_before
        row_pad = (pad_row_before, pad_row_after)
    elif row_diff < 0:  # Need row cropping
        row_start = (-row_diff) // 2
        row_crop = (row_start, rows_o)
    # else: row_diff == 0, no change needed
    
    # Handle columns
    if col_diff > 0:  # Need column padding
        pad_col_before = col_diff // 2
        pad_col_after = col_diff - pad_col_before
        col_pad = (pad_col_before, pad_col_after)
    elif col_diff < 0:  # Need column cropping
        col_start = (-col_diff) // 2
        col_crop = (col_start, cols_o)
    # else: col_diff == 0, no change needed
    
    # Apply row operations
    if row_pad is not None:
        FLAIR_array = np.pad(FLAIR_array, 
                            ((0, 0), (row_pad[0], row_pad[1]), (0, 0)),
                            mode='constant', constant_values=0)
        if two_modalities:
            T1_array = np.pad(T1_array, 
                             ((0, 0), (row_pad[0], row_pad[1]), (0, 0)),
                             mode='constant', constant_values=0)
    elif row_crop is not None:
        FLAIR_array = FLAIR_array[:, row_crop[0]:row_crop[0]+rows_standard, :]
        if two_modalities:
            T1_array = T1_array[:, row_crop[0]:row_crop[0]+rows_standard, :]
    
    # Apply column operations
    if col_pad is not None:
        FLAIR_array = np.pad(FLAIR_array, 
                            ((0, 0), (0, 0), (col_pad[0], col_pad[1])),
                            mode='constant', constant_values=0)
        if two_modalities:
            T1_array = np.pad(T1_array, 
                             ((0, 0), (0, 0), (col_pad[0], col_pad[1])),
                             mode='constant', constant_values=0)
    elif col_crop is not None:
        FLAIR_array = FLAIR_array[:, :, col_crop[0]:col_crop[0]+cols_standard]
        if two_modalities:
            T1_array = T1_array[:, :, col_crop[0]:col_crop[0]+cols_standard]
    
    # Package metadata
    transform_info = {
        'row_pad': row_pad,
        'col_pad': col_pad,
        'row_crop': row_crop,
        'col_crop': col_crop,
        'original_shape': (rows_o, cols_o)
    }
    
    if two_modalities:
        imgs_two_channels = np.concatenate((FLAIR_array[..., np.newaxis], T1_array[..., np.newaxis]), axis=3)
        return imgs_two_channels, transform_info
    else: 
        return FLAIR_array[..., np.newaxis], transform_info


def postprocessing(FLAIR_array, pred, transform_info):
    start_slice = int(np.shape(FLAIR_array)[0]*per)
    num_o = np.shape(FLAIR_array)[0]
    rows_o, cols_o = transform_info['original_shape']
    
    # Start with the prediction
    original_pred = pred[:, :, :, 0].copy()
    
    # Reverse row operations
    if transform_info['row_pad'] is not None:
        # We padded, so crop back
        pad_before, pad_after = transform_info['row_pad']
        original_pred = original_pred[:, pad_before:rows_standard-pad_after, :]
    elif transform_info['row_crop'] is not None:
        # We cropped, so pad back
        row_start, _ = transform_info['row_crop']
        temp = np.zeros((original_pred.shape[0], rows_o, original_pred.shape[2]), dtype=np.float32)
        temp[:, row_start:row_start+rows_standard, :] = original_pred
        original_pred = temp
    
    # Reverse column operations
    if transform_info['col_pad'] is not None:
        # We padded, so crop back
        pad_before, pad_after = transform_info['col_pad']
        original_pred = original_pred[:, :, pad_before:cols_standard-pad_after]
    elif transform_info['col_crop'] is not None:
        # We cropped, so pad back
        col_start, _ = transform_info['col_crop']
        temp = np.zeros((original_pred.shape[0], original_pred.shape[1], cols_o), dtype=np.float32)
        temp[:, :, col_start:col_start+cols_standard] = original_pred
        original_pred = temp
    
    # Zero out top and bottom slices
    original_pred[0:start_slice, ...] = 0
    original_pred[(num_o-start_slice):num_o, ...] = 0
    
    return original_pred



## some pre-defined parameters 
rows_standard = 200  # the input size. fixed.
cols_standard = 200 # fixed.
thresh = 30   # threshold for getting the brain mask. fixed.
per = 0.125 # simply filter false postives. fixed. 

two_modalities = True  # if it's FLAIR-only, please set it to False.

inputDir = 'input_dir'
outputDir = 'output_dir'
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

# Read data----------------------------------------------------------------------------
subject_list = os.listdir(inputDir)
for ss in subject_list:
    print(ss)
    if not os.path.exists(os.path.join(outputDir, ss)):
        os.mkdir(os.path.join(outputDir, ss))
        os.mkdir(os.path.join(outputDir, ss, 'derivatives'))
    if two_modalities:
        img_shape = (rows_standard, cols_standard, 2)
        model_dir = 'models/FLAIR_T1'
        FLAIR_image = sitk.ReadImage(os.path.join(inputDir, ss, 'anat', ss+'_FLAIR.nii.gz'))
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        T1_image = sitk.ReadImage(os.path.join(inputDir, ss, 'anat', ss+'_T1.nii.gz'))
        T1_array = sitk.GetArrayFromImage(T1_image)
        imgs_test, transform_info = preprocessing(np.float32(FLAIR_array), np.float32(T1_array)) 
    else:
        img_shape = (rows_standard, cols_standard, 1)
        model_dir = 'models/FLAIR_only'
        FLAIR_image = sitk.ReadImage(os.path.join(inputDir, ss, 'anat', ss+'_FLAIR.nii.gz'))
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        T1_array = []  # Fixed: was T3_array
        imgs_test, transform_info = preprocessing(np.float32(FLAIR_array), np.float32(T1_array)) 


    # Load model---------------------------------------------
    model = get_unet(img_shape) 
    model.load_weights(os.path.join(model_dir, '0.h5'))  # 3 ensemble models
    print('-'*30)
    print('Predicting segmentation masks on test data...') 
    pred_1 = model.predict(imgs_test, batch_size=1, verbose=1)
    model.load_weights(os.path.join(model_dir, '1.h5')) 
    pred_2 = model.predict(imgs_test, batch_size=1, verbose=1)
    model.load_weights(os.path.join(model_dir, '2.h5'))
    pred_3 = model.predict(imgs_test, batch_size=1, verbose=1)
    pred = (pred_1 + pred_2 + pred_3) / 3
    pred[pred[..., 0] > 0.45] = 1      # 0.45 as the threshold
    pred[pred[..., 0] <= 0.45] = 0

    original_pred = postprocessing(FLAIR_array, pred, transform_info)  # get the original size to match

    # Save data-------------------------------------------------------
    filename_resultImage = os.path.join(outputDir, ss, 'derivatives', ss+'_seg.nii.gz')  # Fixed: was 'derelatives'
    pred_image = (sitk.GetImageFromArray(original_pred))
    pred_image.CopyInformation(FLAIR_image)
    sitk.WriteImage(pred_image, filename_resultImage)
