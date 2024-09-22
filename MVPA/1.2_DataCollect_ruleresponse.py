# MVPA; python3.10
# 2024/09/10 by Chuanyong Xu (refer to ning mei. @author: nmei; Created on Mon Apr 17 11:46:05 2023)
import os
import gc
import nilearn
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib

from matplotlib import pyplot as plt
from PIL import Image
from IPython import display
from joblib import Parallel, delayed

from scipy.spatial import distance
from scipy.stats import zscore
from nibabel import load as load_fmri
from scipy.stats import spearmanr

from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker
from nilearn.image import new_img_like
from nilearn.image import resample_to_img
from nilearn.image import load_img
from nilearn import plotting, image
from nilearn.datasets import load_mni152_template

sns.set_context('poster')
sns.set_style('white')

#==============================================================================
# seeting
exp_type = 'inference'
data_dir = '/home/xuchuanyong/Documents/DATA_analysis/MRI_SZU_HR'
# ------------------------------------------------------------------------------------------
subs = [1101, 1103, 1104, 1105, 1108, 1110,1111, 1113, 1114, 1116, 1117, 1120, 1121, 1122, 1123, 1124,
        1125, 1126, 1128, 1129, 1130, 1132]################# change*
condition = 'rule_respons'################# change*
conditions_f = 'data_raw_decodingRSA_ruleresponse_'################# change*
results_dir = 'data_rule_respons'################# change*
# ------------------------------------------------------------------------------------------
for sub in subs:
    sub_csv = sub
    sub = f'sub{sub:03d}'
    bold_files = np.sort(glob(os.path.join(data_dir,'preprocess_by_spm12','fun_s1_4d0*', f'{sub}', 'swrafun_4D.nii'))) 
    csv_files = os.path.join(data_dir,'preprocess_by_spm12','decoding_rsa_behaviors', f'{conditions_f}' f'{sub_csv}' '.csv') 
    #example_fun = os.path.join(data_dir,'example_fun.nii.gz')
    img_hdr=nib.load(os.path.join(data_dir,'preprocess_by_spm12','BrainMask_05_91x109x91.hdr'))#covert hdr to nii
    nib.save(img_hdr,os.path.join(data_dir,'preprocess_by_spm12','BrainMask_05_91x109x91.nii'))#save coverted nii
    mask_img = os.path.join(data_dir,'preprocess_by_spm12','BrainMask_05_91x109x91.nii') #load mask
    del img_hdr
    gc.collect()
    
    #==============================================================================
    # import chardet # for check out the type of file
    csv_data = []
    csv_data = pd.read_csv(csv_files)
    print('csv_data load done')
    
    # load bold data and csv
    sub_dir = f'/home/xuchuanyong/Documents/DATA_analysis/MRI_SZU_HR/Decoding_mvpa/{results_dir}/{sub}' 
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    csv_data.to_csv(os.path.join(sub_dir, f'events_{condition}.csv'))#save coverted csv 

    bold_data = []
    for bold_file in bold_files:
        nii_bold = nib.load(bold_file)
        masker = NiftiMasker(mask_img = mask_img,
                             standardize=True,
                             high_pass=1/128,
                             t_r=0.85,
                             n_jobs=-1,
                             ) # get the bold data only in brain mask
        nii_bold = masker.fit_transform(nii_bold) # get the bold data only in brain mask
        nii_bold = masker.inverse_transform(nii_bold)
        temp_bold = nii_bold.get_fdata()
        bold_data.append(temp_bold)
    del nii_bold
    gc.collect()
    print('Data in mask extracted done')
    
    bold_data=np.concatenate(bold_data, axis=-1)
    del temp_bold
    gc.collect()
    print('Data concatenate done')
    
    #==============================================================================
    # prepare the whole brain BOLD signal that are averaged from all sessions
    bold_average, temp_condition = [], []
    for _conf, df_sub in csv_data.groupby([condition]): ################# change*
        for index, tr in df_sub.iterrows():
            numbers = pd.DataFrame(tr['time_indices'].split('+'), columns=['numbers']).astype(int)
            temp = bold_data[..., numbers]
            temp = temp[..., 0]
            bold_average.append(temp.mean(-1)) # mean for each trial
    bold_average = np.stack(bold_average)
    
    bold_average = [arr.ravel() for arr in bold_average]
    bold_average = np.vstack(bold_average)
    mask_img_data = masker.mask_img_.get_fdata()
    valid_voxels_mask = mask_img_data>0
    print(np.sum(mask_img_data>0))
    bold_average = bold_average[:,valid_voxels_mask.ravel()] # *******only includ voxels with value > 0
    whole_brain_average = masker.inverse_transform(bold_average)
    BOLD_image = np.asanyarray(whole_brain_average.dataobj)
    print(BOLD_image.shape)
    nib.save(whole_brain_average,os.path.join(sub_dir,'whole_brain_average.nii.gz'))#save coverted nii
    
    nii_3d = image.index_img(whole_brain_average, 22)
    #plotting.plot_anat(nii_3d)
    #plotting.show()
