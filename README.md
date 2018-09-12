# wmh_ibbmTum
winning method for WMH segmentation challenge in MICCAI 2017

Requirement: 
Keras 1.2, Tensorflow, h5py


For the weights of the model we submitted to the challenge, please download them via: https://drive.google.com/drive/folders/1i4Y9M0yW3JN_WC8Fj1VlCdaE2lvG_9Ar 

For the .npy files to run the leave-one-subject-out experiments, please download via: https://drive.google.com/open?id=1m0H9vbFV8yijvuTsAqRAUQGGitanNw_k

Decriptions for the python code:

train_leave_one_out.py: train U-Net models under leave-one-subject-out protocol. For options, you can train models with single modelity or without data augmentation.
test_leave_one_out.py: test U-Net models under leave-one-subject-out protocol. The codes also include the preprocessing of the original data.
evaluation.py: evaluation code provided by the challenge organizor. 

images_three_datasets_sorted.npy: preprocessed dataset including Utrecht, Singapore and GE3T. The order of the patients is sorted.
masks_three_datasets_sorted.npy: preprocessed masks including Utrecht, Singapore and GE3T corresponding to the preprocessed data. The order of the patients is sorted.


