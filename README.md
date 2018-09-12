# wmh_ibbmTum
winning method for WMH segmentation challenge in MICCAI 2017

Requirements: 
Keras 2.0.5, Tensorflow, Python 2.7, h5py


For the weights of the model we submitted to the challenge, please download them via: https://drive.google.com/drive/folders/1i4Y9M0yW3JN_WC8Fj1VlCdaE2lvG_9Ar . You can use these models for segmenting your cases. We also have Docker file to do segmentation if you are interested. Please feel free to contact me.   

For the .npy files to run the leave-one-subject-out experiments, please download via: https://drive.google.com/open?id=1m0H9vbFV8yijvuTsAqRAUQGGitanNw_k .

Decriptions for the python code:

train_leave_one_out.py: train U-Net models under leave-one-subject-out protocol. For options, you can train models with single modelity or without data augmentation.
test_leave_one_out.py: test U-Net models under leave-one-subject-out protocol. The codes also include the preprocessing of the original data.
evaluation.py: evaluation code provided by the challenge organizor. 

images_three_datasets_sorted.npy: preprocessed dataset including Utrecht, Singapore and GE3T. The order of the patients is sorted.
masks_three_datasets_sorted.npy: preprocessed masks including Utrecht, Singapore and GE3T corresponding to the preprocessed data. The order of the patients is sorted.



The detailed description of our method is published in NeuroImage: https://www.sciencedirect.com/science/article/pii/S1053811918305974?via%3Dihub .
Please cite our work if you find the code is useful for your research.

