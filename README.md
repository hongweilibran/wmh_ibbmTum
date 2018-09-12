# wmh_ibbmTum
winning method for WMH segmentation challenge in MICCAI 2017


images_three_datasets_sorted.npy: preprocessed dataset including Utrecht, Singapore and GE3T.
masks_three_datasets_sorted.npy: preprocessed masks including Utrecht, Singapore and GE3T corresponding to the preprocessed data.
(please download via: https://drive.google.com/open?id=1m0H9vbFV8yijvuTsAqRAUQGGitanNw_k)

train_leave_one_out.py: train models under leave-one-subject-out protocol. For options, you can train models with single modelity or without data augmentation.
test_leave_one_out.py: test models under leave-one-subject-out protocol. The codes also include the preprocessing of the original data.
evaluation.py: evaluation code provided by the challenge organizor. 
