#Codes for validating the WMH Challenge training Datasets. The algorithm won the WMH Challenge.
#Codes are written by Mr. Hongwei Li (hongwei.li@tum.de), Mr. Gongfa Jiang and Miss. Zhaolei Wang from Sun Yat-sen University and University of Dundee.
#

from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import difflib
import SimpleITK as sitk
import scipy.spatial
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.optimizers import Adam
from evaluation import getDSC, getHausdorff, getLesionDetection, getAVD, getImages  #please download evaluation.py from the WMH website
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from show import imshow
from scipy import ndimage
#from sklearn.utils import class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go 

### ----define loss function for U-net ------------
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

### ----define U-net architecture--------------
def get_unet(img_shape = None):

        dim_ordering = 'tf'
        inputs = Input(shape = img_shape)
        concat_axis = -1
        ### the size of convolutional kernels is defined here    
        conv1 = Convolution2D(64, 5, 5, activation='relu', border_mode='same', dim_ordering=dim_ordering, name='conv1_1')(inputs)
        conv1 = Convolution2D(64, 5, 5, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering)(conv1)
        conv2 = Convolution2D(96, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(pool1)
        conv2 = Convolution2D(96, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering)(conv2)

        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(pool2)
        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering)(conv3)

        conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(pool3)
        conv4 = Convolution2D(256, 4, 4, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering)(conv4)

        conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(pool4)
        conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv5)

        up_conv5 = UpSampling2D(size=(2, 2), dim_ordering=dim_ordering)(conv5)
        ch, cw = get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch,cw), dim_ordering=dim_ordering)(conv4)
        up6 = merge([up_conv5, crop_conv4], mode='concat', concat_axis=concat_axis)
        conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(up6)
        conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv6)

        up_conv6 = UpSampling2D(size=(2, 2), dim_ordering=dim_ordering)(conv6)
        ch, cw = get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch,cw), dim_ordering=dim_ordering)(conv3)
        up7 = merge([up_conv6, crop_conv3], mode='concat', concat_axis=concat_axis)
        conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(up7)
        conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv7)

        up_conv7 = UpSampling2D(size=(2, 2), dim_ordering=dim_ordering)(conv7)
        ch, cw = get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch,cw), dim_ordering=dim_ordering)(conv2)
        up8 = merge([up_conv7, crop_conv2], mode='concat', concat_axis=concat_axis)
        conv8 = Convolution2D(96, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(up8)
        conv8 = Convolution2D(96, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv8)

        up_conv8 = UpSampling2D(size=(2, 2), dim_ordering=dim_ordering)(conv8)
        ch, cw = get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch,cw), dim_ordering=dim_ordering)(conv1)
        up9 = merge([up_conv8, crop_conv1], mode='concat', concat_axis=concat_axis)
        conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(up9)
        conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv9)

        ch, cw = get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=(ch, cw), dim_ordering=dim_ordering)(conv9)
        conv10 = Convolution2D(1, 1, 1, activation='sigmoid', dim_ordering=dim_ordering)(conv9)
        model = Model(input=inputs, output=conv10)
        model.compile(optimizer=Adam(lr=(1e-4)*2), loss=dice_coef_loss, metrics=[dice_coef_for_training])

        return model

###----define prepocessing methods/tricks for different datasets------------------------
def Utrecht_preprocessing(FLAIR_image, T1_image):

    channel_num = 2
    print(np.shape(FLAIR_image))
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    T1_image = np.float32(T1_image)

    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
    imgs_mask_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard,1), dtype=np.float32)

    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
    brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
    for iii in range(np.shape(FLAIR_image)[0]):
        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
    FLAIR_image = FLAIR_image[:, (image_rows_Dataset/2-rows_standard/2):(image_rows_Dataset/2+rows_standard/2), (image_cols_Dataset/2-cols_standard/2):(image_cols_Dataset/2+cols_standard/2)]
    brain_mask_FLAIR = brain_mask_FLAIR[:, (image_rows_Dataset/2-rows_standard/2):(image_rows_Dataset/2+rows_standard/2), (image_cols_Dataset/2-cols_standard/2):(image_cols_Dataset/2+cols_standard/2)]
    ###------Gaussion Normalization here
    FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
    FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])
    # T1 -----------------------------------------------
    brain_mask_T1[T1_image >=thresh_T1] = 1
    brain_mask_T1[T1_image < thresh_T1] = 0
    for iii in range(np.shape(T1_image)[0]):
        brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
    T1_image = T1_image[:, (image_rows_Dataset/2-rows_standard/2):(image_rows_Dataset/2+rows_standard/2), (image_cols_Dataset/2-cols_standard/2):(image_cols_Dataset/2+cols_standard/2)]
    brain_mask_T1 = brain_mask_T1[:, (image_rows_Dataset/2-rows_standard/2):(image_rows_Dataset/2+rows_standard/2), (image_cols_Dataset/2-cols_standard/2):(image_cols_Dataset/2+cols_standard/2)]
    #------Gaussion Normalization
    T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      
    T1_image /=np.std(T1_image[brain_mask_T1 == 1])
    #---------------------------------------------------
    FLAIR_image  = FLAIR_image[..., np.newaxis]
    T1_image  = T1_image[..., np.newaxis]
    imgs_two_channels = np.concatenate((FLAIR_image, T1_image), axis = 3)
    print(np.shape(imgs_two_channels))
    return imgs_two_channels

def Utrecht_postprocessing(FLAIR_array, pred):
    start_slice = 6
    num_selected_slice = np.shape(FLAIR_array)[0]
    image_rows_Dataset = np.shape(FLAIR_array)[1]
    image_cols_Dataset = np.shape(FLAIR_array)[2]
    original_pred = np.ndarray(np.shape(FLAIR_array), dtype=np.float32)
    original_pred[:,(image_rows_Dataset-rows_standard)/2:(image_rows_Dataset+rows_standard)/2,(image_cols_Dataset-cols_standard)/2:(image_cols_Dataset+cols_standard)/2] = pred[:,:,:,0]
    
    original_pred[0:start_slice, :, :] = 0
    original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
    return original_pred



def GE3T_preprocessing(FLAIR_image, T1_image):

  #  start_slice = 10
    channel_num = 2
    start_cut = 46
    print(np.shape(FLAIR_image))
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    FLAIR_image = np.float32(FLAIR_image)
    T1_image = np.float32(T1_image)

    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
    imgs_mask_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard,1), dtype=np.float32)
    FLAIR_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
    T1_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)

    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
    brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
    for iii in range(np.shape(FLAIR_image)[0]):
  
        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
        #------Gaussion Normalization
    FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
    FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])

    FLAIR_image_suitable[...] = np.min(FLAIR_image)
    FLAIR_image_suitable[:, :, (cols_standard/2-image_cols_Dataset/2):(cols_standard/2+image_cols_Dataset/2)] = FLAIR_image[:, start_cut:start_cut+rows_standard, :]
   
    # T1 -----------------------------------------------
    brain_mask_T1[T1_image >=thresh_T1] = 1
    brain_mask_T1[T1_image < thresh_T1] = 0
    for iii in range(np.shape(T1_image)[0]):
 
        brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
        #------Gaussion Normalization
    T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      #Gaussion Normalization
    T1_image /=np.std(T1_image[brain_mask_T1 == 1])

    T1_image_suitable[...] = np.min(T1_image)
    T1_image_suitable[:, :, (cols_standard-image_cols_Dataset)/2:(cols_standard+image_cols_Dataset)/2] = T1_image[:, start_cut:start_cut+rows_standard, :]
    #---------------------------------------------------
    FLAIR_image_suitable  = FLAIR_image_suitable[..., np.newaxis]
    T1_image_suitable  = T1_image_suitable[..., np.newaxis]
    
    imgs_two_channels = np.concatenate((FLAIR_image_suitable, T1_image_suitable), axis = 3)
    print(np.shape(imgs_two_channels))
    return imgs_two_channels

def GE3T_postprocessing(FLAIR_array, pred):
    start_slice = 11
    start_cut = 46
    num_selected_slice = np.shape(FLAIR_array)[0]
    image_rows_Dataset = np.shape(FLAIR_array)[1]
    image_cols_Dataset = np.shape(FLAIR_array)[2]
    original_pred = np.ndarray(np.shape(FLAIR_array), dtype=np.float32)
    original_pred[:, start_cut:start_cut+rows_standard,:] = pred[:,:, (rows_standard-image_cols_Dataset)/2:(rows_standard+image_cols_Dataset)/2,0]

    original_pred[0:start_slice, :, :] = 0
    original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
    return original_pred



###---Here comes the main funtion--------------------------------------------
###---Leave one patient out validation--------------------------------------------

patient_num = 60
patient_count = 0
rows_standard = 200
cols_standard = 200
thresh_FLAIR = 70      #to mask the brain
thresh_T1 = 30
para_array = [[0.958, 0.958, 3], [1.00, 1.00, 3], [1.20, 0.977, 3]]    # parameters of the scanner
para_array = np.array(para_array, dtype=np.float32)

	
#read the dirs of test data 
input_dir_1 = 'raw/Utrecht'
input_dir_2 = 'raw/Singapore'
input_dir_3 = 'raw/GE3T'
###---dir to save results---------
outputDir = 'evaluation_result_LOOV'
#-------------------------------------------
dirs = os.listdir(input_dir_1) + os.listdir(input_dir_2) + os.listdir(input_dir_3)
#All the slices and the corresponding patients id
imgs_three_datasets_two_channels = np.load('imgs_three_datasets_two_channels.npy')
imgs_mask_three_datasets_two_channels = np.load('imgs_mask_three_datasets_two_channels.npy')
slices_patient_id_label = np.load('slices_patient_id_label.npy')


for dir_name in dirs:
	print('dir_name is:')
	print(dir_name)
	if patient_count < 20:
		inputDir = input_dir_1
	elif patient_count > 19 and patient_count < 40:
		inputDir = input_dir_2
	elif patient_count > 39:
		inputDir = input_dir_3
	FLAIR_image = sitk.ReadImage(os.path.join(inputDir, dir_name, 'pre', 'FLAIR.nii.gz'))
	T1_image = sitk.ReadImage(os.path.join(inputDir, dir_name, 'pre', 'T1.nii.gz'))
	FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
	T1_array = sitk.GetArrayFromImage(T1_image)
	#Proccess testing data-----
	para_FLAIR = np.ndarray((1,3), dtype=np.float32)
	para_FLAIR_ = FLAIR_image.GetSpacing()
	para_FLAIR[0,0] = round(para_FLAIR_[0],3)   # get spacing parameters of the data
	para_FLAIR[0,1] = round(para_FLAIR_[1],3)  
	para_FLAIR[0,2] = round(para_FLAIR_[2],3) 
	if np.array_equal(para_FLAIR[0], para_array[0]) :
		print('From Utrecht!')
		imgs_test = Utrecht_preprocessing(FLAIR_array, T1_array)
	elif np.array_equal(para_FLAIR[0], para_array[1]):
		print('From Singapore!')
		imgs_test = Utrecht_preprocessing(FLAIR_array, T1_array)
	elif np.array_equal(para_FLAIR[0], para_array[2]):
		print('From GE3T!')
	 	imgs_test = GE3T_preprocessing(FLAIR_array, T1_array)

	###---train u-net models-------------------------------------------------------------------------------
	training_index = slices_patient_id_label != patient_count
	test_index = slices_patient_id_label == patient_count
	dim_training = sum(training_index)
	dim_test = sum(test_index)
	print('the dim of training set:')
	print(dim_training[0])
	imgs_train = np.ndarray((dim_training[0], rows_standard, cols_standard, 2), dtype=np.float32)
	imgs_test_selected = np.ndarray((dim_test[0], rows_standard, cols_standard, 2), dtype=np.float32)

	imgs_mask_train = np.ndarray((dim_training[0], rows_standard, cols_standard, 1), dtype=np.float32)
	imgs_mask_test_selected = np.ndarray((dim_test[0], rows_standard, cols_standard, 1), dtype=np.float32)
	count_index_train = 0
	count_index_test = 0
	for iii in range(training_index.shape[0]):
		if training_index[iii] == 1:
			imgs_train[count_index_train, ...] = imgs_three_datasets_two_channels[iii, ...]
			imgs_mask_train[count_index_train, ...] = imgs_mask_three_datasets_two_channels[iii, ...]
			count_index_train = count_index_train + 1
		if training_index[iii] == 0:
			imgs_test_selected[count_index_test, ...] = imgs_three_datasets_two_channels[iii, ...]
			imgs_mask_test_selected[count_index_test, ...] = imgs_mask_three_datasets_two_channels[iii, ...]
			count_index_test = count_index_test + 1
	
	print('training dataset dimension:')
	print(imgs_train.shape[0])
	img_shape=(rows_standard, cols_standard, 2)
	
	
	print('-'*30)
	print('Fitting model...')
	print('-'*30)
    ###---parameters of training are set here------------------------------------
	ensemble_parameter = 3
	model = get_unet(img_shape)
	pred = model.predict(imgs_test, batch_size=1,verbose=1)
	pred = np.ndarray(np.shape(pred), dtype=np.float32)
	###---ensemble model --------------------------
	for iiii in range(ensemble_parameter):
		model = get_unet(img_shape)
		model_checkpoint = ModelCheckpoint('weights_three_datasets_two_channels_LOOV_'+str(iiii)+'.h5', save_best_only=False, period = 2)
		#model.fit(imgs_train, imgs_mask_train, batch_size=30, nb_epoch= 5, verbose=1, shuffle=True, validation_split=0.0, callbacks=[model_checkpoint])
		#model.save('weights_three_datasets_two_channels_LOOV_'+str(iiii)+'.h5')
		model.load_weights('weights_three_datasets_two_channels_LOOV_'+str(iiii)+'.h5')
		pred_temp = model.predict(imgs_test, batch_size=1,verbose=1)
		pred = pred + pred_temp
	pred = pred/int(ensemble_parameter)
	pred[pred[...,0] > 0.5] = 1      #thresholding 
	pred[pred[...,0] <= 0.5] = 0


	#Postprocessing
	if np.array_equal(para_FLAIR[0], para_array[0]): 		
		print('Utrecht!')
		original_pred = Utrecht_postprocessing(FLAIR_array, pred)
	elif np.array_equal(para_FLAIR[0], para_array[1]):
		print('Singapore!')
		original_pred = Utrecht_postprocessing(FLAIR_array, pred)
	elif np.array_equal(para_FLAIR[0], para_array[2]):
		print('GE3T!')
		original_pred = GE3T_postprocessing(FLAIR_array, pred)
	
	if not os.path.exists(outputDir):
		os.mkdir(outputDir)
	filename_resultImage = os.path.join(outputDir, 'result.nii.gz')
	sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )
	filename_testImage = os.path.join(inputDir, dir_name, 'wmh.nii.gz')
	testImage, resultImage = getImages(filename_testImage, filename_resultImage)
	dsc = getDSC(testImage, resultImage)
	avd = getAVD(testImage, resultImage) 
	h95 = getHausdorff(testImage, resultImage)
	recall, f1 = getLesionDetection(testImage, resultImage)
	print('Result of patient '+str(patient_count))    
	print('Dice',                dsc,       '(higher is better, max=1)')
	print('HD',                  h95, 'mm',  '(lower is better, min=0)')
	print('AVD',                 avd,  '%',  '(lower is better, min=0)')
	print('Lesion detection', recall,       '(higher is better, max=1)')
	print('Lesion F1',            f1,       '(higher is better, max=1)')
#Save result-------------------------------------------------------	
	result_output_dir = os.path.join(outputDir,dir_name)  #directory for images
	if not os.path.exists(result_output_dir):
		os.mkdir(result_output_dir)
	np.save(os.path.join(result_output_dir,'dsc.npy'), dsc)
	np.save(os.path.join(result_output_dir,'avd.npy'), avd)
	np.save(os.path.join(result_output_dir,'h95.npy'), h95)
	np.save(os.path.join(result_output_dir,'recall.npy'), recall)
	np.save(os.path.join(result_output_dir,'f1.npy'), f1)
#-------------------------------------------------------------------
	for iii in range(np.shape(imgs_test_selected)[0]):
		print('saving image:'+ str(iii)+' of patient '+str(patient_count))
		pos_index = np.int32(np.where(imgs_mask_test_selected[iii]==1))
		pred_pos_index = np.array(np.where(pred[iii]>.9))
		insect_index = np.int32(np.where(np.logical_and(pred[iii]>.5, imgs_mask_test_selected[iii]==1)))
		pred_mask = np.zeros([imgs_mask_test_selected[iii].shape[0], imgs_mask_test_selected[iii].shape[1], 3], dtype=np.float32)
		pred_mask[pos_index[0], pos_index[1], 0] = 1.
		pred_mask[pred_pos_index[0], pred_pos_index[1], ...] = 1.
		pred_mask[insect_index[0], insect_index[1], 0] = 0.
		pred_mask[insect_index[0], insect_index[1], 1] = 1.
		pred_mask[insect_index[0], insect_index[1], 2] = 0.
		
		seg_img = np.copy(imgs_test_selected[iii,:,:,0:1]) 	
		seg_img[imgs_mask_test_selected[iii]==1] = np.max(seg_img)
		imshow(imgs_test_selected[iii,:,:,0],seg_img[:,:,0],pred_mask,pred[iii,:,:,0],title=['test image','ground truth','prediction','heat map'])
		plt.savefig(os.path.join(result_output_dir, str(iii) + '_pred.png'))	

#----------------------------------------------------------------------
	patient_count = patient_count+1





















