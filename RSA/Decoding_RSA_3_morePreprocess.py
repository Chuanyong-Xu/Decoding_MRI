# RSA; python3.10
# try:
#     !git clone https://github.com/nmningmei/METASEMA_encoding_model.git
# except Exception:
#     pass

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
from scipy.stats import pearsonr

from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker
from nilearn.image import new_img_like
from nilearn.image import resample_to_img
from nilearn.image import load_img
from nilearn import plotting, image
from numba import njit, prange
#from nilearn.datasets import load_mni152_template

#------------------------------define function 1-------------------------------
def normalize(data, axis=1):
    return data - data.mean(axis).reshape(-1,1)
def sfn(l, msk, myrad, bcast_var):
    BOLD = l[0][msk, :].T.copy()
    model = bcast_var.copy()
    RDM_x = distance.pdist(normalize(BOLD), 'euclidean') # euclidean is better for this dataset?
    RDM_y = distance.pdist(normalize(model), 'euclidean') # euclidean is better for this dataset?
    #D,p = spearmanr(RDM_x, RDM_y)
    D,p = pearsonr(RDM_x, RDM_y) #************************************
    return D

@njit(parallel=True)
def detec_outliers_and_interpolate(data):
    n_timerpoints, n_voxels = data.shape
    for voxel in prange(n_voxels):
        y=data[:,voxel]
        outliers=np.abs(y)>=3
        y[outliers]=np.nan
        
        nans=np.isnan(y)
        if np.all(nans):
            continue
        not_nans=~nans
        x=np.arange(n_timerpoints)
        y[nans]=np.interp(x[nans],x[not_nans],y[not_nans])
        data[:,voxel]=y
    return data

#------------------------------define function 2-------------------------------
def correlation_within_sphere(BOLD_signals,
                              embedding_vectors,
                              row,
                              ):
    RDM_bold = distance.pdist(BOLD_signals[:, row], 'euclidean') # euclidean is better for this dataset?
    RDM_model = distance.pdist(embedding_vectors, 'euclidean') # euclidean is better for this dataset?
    D = 1 - distance.cdist([RDM_bold],[RDM_model],'correlation')[0][0]
    # D,p = spearmanr(RDM_bold, RDM_model) #************************************
    
    #D,p = pearsonr(RDM_bold, RDM_model) #************************************
    #D = distance.euclidean(RDM_bold, RDM_model)
    return D

#------------------------------define function 3-------------------------------
from nilearn import masking
from nilearn.image.resampling import coord_transform
try:
    from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity
except:
    from nilearn.maskers.nifti_spheres_masker import _apply_mask_and_get_affinity
def get_sphere_data(mask, BOLD_data, radius,):
    if type(BOLD_data) == str:
        BOLD_data = nilearn.image.load_img(BOLD_data)
    process_mask, process_mask_affine = masking.load_mask_img(mask)
    process_mask_coords = np.where(process_mask !=0)
    process_mask_coords = coord_transform(process_mask_coords[0],
                                          process_mask_coords[1],
                                          process_mask_coords[2],
                                          process_mask_affine,
                                          )
    process_mask_coords = np.asarray(process_mask_coords).T
    
    X, A = _apply_mask_and_get_affinity(seeds=process_mask_coords,
                                        niimg=BOLD_data,
                                        radius=radius,
                                        allow_overlap=True,
                                        mask_img = mask,
                                        )
    return X, A, process_mask, process_mask_affine, process_mask_coords
#------------------------------------------------------------------------------

cond_cat = 'type_dev_error_4'
cond_value = 'confidence'
conditions_f = 'data_raw_decodingRSA_confidence_'
condition = 'inference'
data_dir_ori = '/home/xuchuanyong/Documents/DATA_analysis/MRI_SZU_HR'
group_means_arry_all = []# plotting all subs RDM
subs=[1101, 1103, 1104, 1105, 1108, 1110, 1111, 1113, 1114, 1116, 1117, 1120,
      1121, 1122, 1123, 1124, 1125, 1126, 1128, 1129, 1130, 1132] # 

for sub in subs:
    sub_csv = sub
    sub = f'sub{sub:03d}'
    #==========================================================================
    # setting
    bold_files = np.sort(glob(os.path.join(data_dir_ori,'preprocess_by_spm12','fun_s1_4d0*',f'{sub}','swrafun_4D.nii')))
    rp_files = np.sort(glob(os.path.join(data_dir_ori,'preprocess_by_spm12','fun_s1_4d0*',f'{sub}','rp_afun_4D.txt')))
    csv_files = os.path.join(data_dir_ori,'preprocess_by_spm12','decoding_rsa_behaviors', f'{conditions_f}' f'{sub_csv}' '.csv') 
    #example_fun = os.path.join(data_dir_ori,'preprocess_by_spm12','example_fun.nii.gz')
    #img_hdr=nib.load(os.path.join(data_dir_ori,'preprocess_by_spm12','BrainMask_05_91x109x91.hdr'))#covert hdr to nii
    #nib.save(img_hdr,os.path.join(data_dir_ori,'preprocess_by_spm12','BrainMask_05_91x109x91.nii'))#save coverted nii
    mask_img = os.path.join(data_dir_ori,'preprocess_by_spm12','BrainMask_05_91x109x91.nii') #load mask
    #del img_hdr
    gc.collect()
    #
    data_dir = f'/{data_dir_ori}/Decoding_RSA/{cond_cat}/data/{sub}'
    results_dir = f'{data_dir_ori}/Decoding_RSA/{cond_cat}/results'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    #==========================================================================
    # import chardet # for check out the type of file
    csv_data = []
    #for csv_file in csv_files:
    temp_csv = pd.read_csv(csv_files)
    csv_data.append(temp_csv)
    csv_data = pd.concat(csv_data)
    print(csv_data)
    csv_data.to_csv(os.path.join(data_dir, f'events_{cond_cat}.csv'))#save coverted csv 

    # plot the corr map
    #csv_data[cond_value] = np.where(csv_data[cond_value] == 0, csv_data[cond_value] + np.random.normal(0, 1e-6, len(csv_data[cond_value] )),csv_data[cond_value])
    #scaler = StandardScaler()
    # csv_data[cond_value] = scaler.fit_transform(csv_data[cond_value].values.reshape(-1,1))
    csv_data[cond_value] = csv_data[cond_value].apply(lambda x: np.log(x + 1e-16))
    group_means = csv_data.groupby(cond_cat)[cond_value].mean()
    group_means_arry = group_means.values.reshape(-1, 1)
    
    # group_means_arry_all.append(group_means_arry)# plotting all subs RDM
    # #----------------
    # group_means_arry_all=np.hstack(group_means_arry_all)# plotting all subs RDM
    
    corr_matrix = distance.squareform(distance.pdist(group_means_arry, metric='euclidean')) #using the mean of each var to calculate euclidean
    csv_data.to_csv(os.path.join(results_dir, f'events_{cond_cat}_{sub}.csv'))#save coverted csv 
    
    sns.set_context('poster')
    sns.set_style('white')
    plt.figure(figsize = (6, 5))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', square=True)
    plt.show()

    grouped = csv_data.groupby(cond_cat)[cond_value]
    df_plot = pd.concat([grouped.get_group(i).reset_index(drop=True) for i in sorted(csv_data[cond_cat].unique())],axis=1)
    #########################################################wait for revise
    df_plot = np.reshape(np.nanmean(df_plot, axis=0), (1,9)) #using the mean of each var to calculate euclidean later
    df_plot = pd.DataFrame(df_plot, columns=['1', '2', '3', '4',
                                             '5', '6', '7', '8', '9'])
    
    
    #============================================================================== 
    masker = NiftiMasker(mask_img = mask_img,
                         detrend=True,
                         high_pass=1/128,
                         t_r=0.85,
                         standardize=True,
                         n_jobs=-1,
                         ).fit() # get the bold data only in brain mask
    if os.path.exists(os.path.join(data_dir,'whole_brain_average.nii.gz')):
        whole_brain_average = nib.load(os.path.join(data_dir,'whole_brain_average.nii.gz'))
        conf_average=list(('1', '2', '3', '4','5', '6', '7', '8', '9'))
    else: 
    # load bold data and csv
        bold_data = []
        for bold_file, rp_file in zip(bold_files, rp_files):
            nii_bold = nib.load(bold_file)
            nii_bold = masker.transform(nii_bold)
            #----------------       
            #confounds = np.loadtxt(rp_file) #head motions parameters
            #nii_bold = masker.transform(nii_bold, confounds=confounds) # get the bold data only in brain mask
            #nii_bold=detec_outliers_and_interpolate(nii_bold)
            #----------------
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
            # affine = nii_bold.affine # save numpy file as nii
            # whole_brain_data_nii = nib.Nifti1Image(whole_brain_data, affine)
            # nib.save(whole_brain_data_nii,os.path.join(data_dir,'whole_brain_data.nii.gz'))#save coverted nii
            #==============================================================================
        # prepare the whole brain BOLD signal that are averaged from all sessions
        bold_average, conf_average = [], []
        for _conf, df_sub in csv_data.groupby([cond_cat]):
            temp_condition = []
            for index, tr in df_sub.iterrows():
                numbers = pd.DataFrame(tr['time_indices'].split('+'), columns=['numbers']).astype(int)
                temp = bold_data[..., numbers]
                temp = temp[..., 0]
                temp_condition.append(temp.mean(-1)) # mean for each trial
            temp_condition = np.stack(temp_condition)
            bold_average.append(temp_condition.mean(0)) # mean for each condition
            conf_average.extend(map(str, df_sub[cond_cat].unique().tolist()))
        
        bold_average = [arr.ravel() for arr in bold_average]
        bold_average = np.vstack(bold_average)
        mask_img_data = masker.mask_img_.get_fdata()
        valid_voxels_mask = mask_img_data>0
        print(np.sum(mask_img_data>0))
        bold_average = bold_average[:,valid_voxels_mask.ravel()] # *******only includ voxels with value > 0   
        whole_brain_average = masker.inverse_transform(bold_average)
        BOLD_image = np.asanyarray(whole_brain_average.dataobj)
        print(BOLD_image.shape)
        # nii_3d = image.index_img(whole_brain_average, 22)
        # plotting.plot_anat(nii_3d)
        # plotting.show() 
        nib.save(whole_brain_average, os.path.join(data_dir,'whole_brain_average.nii.gz'))
        np.save(os.path.join(data_dir,'bold_average.npy'), bold_average, allow_pickle=True)
        #------------------------------RSA: searchlight---------------------------------
    radius = 6 # in mm
    # ---------
    if os.path.exists(os.path.join(data_dir, 'rows.npy')):
        rows = np.load(os.path.join(data_dir, 'rows.npy'), allow_pickle=True)
    else: 
        X, A, process_mask, process_mask_affine, process_mask_coords = get_sphere_data(mask_img,
                                                                                       whole_brain_average,
                                                                                       radius = radius,
                                                                                       )
        rows = A.rows
        np.save(os.path.join(data_dir,'rows.npy'), rows, allow_pickle=True)

    # ---------run the searchlight using parallelization
    bold_average = np.load(os.path.join(data_dir, 'bold_average.npy'), allow_pickle=True)
    correlations = Parallel(n_jobs = -1,
                            verbose=1)(delayed(correlation_within_sphere)(**{
                                'BOLD_signals':bold_average,
                                'embedding_vectors':df_plot[conf_average].values.T,
                                'row':row}) for row in rows)
    correlations = np.array(correlations)
    
    correlations = masker.inverse_transform(correlations)
    nib.save(correlations, os.path.join(results_dir, f'correlations_{cond_cat}_{sub}.nii.gz'))#save coverted csv
    
    # ---------visualization
    plotting.plot_stat_map(correlations,
                           mask_img,
                           threshold = 1e-3,
                           draw_cross = False,
                           cmap = plt.cm.coolwarm,
                           vmax = 1,
                           )
    plotting.show()

##------------------------------RSA: group test---------------------------------

