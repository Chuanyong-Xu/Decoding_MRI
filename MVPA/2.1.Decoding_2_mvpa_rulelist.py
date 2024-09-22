# MVPA; python3.10
# 2024/09/10 by Chuanyong Xu (refer to ning mei. @author: nmei; Created on Mon Apr 17 11:46:05 2023)
# rule_list, rule_response, difficulties, confidence, rule_swirch
import nilearn
import os,gc
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob
from nilearn.maskers import NiftiMasker
from nilearn.image import load_img
from nilearn import plotting, image
from matplotlib import pyplot as plt
#from utils import get_sphere_data,concept_mapping
from sklearn.model_selection import check_cv,cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import ttest_1samp
try:
    from sklearn.utils.testing import ignore_warnings
    from sklearn.exceptions import ConvergenceWarning
except:
    from sklearn.utils._testing import ignore_warnings
    from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel,delayed

#==============================================================================
#------------------------------define function 1-------------------------------
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

# ---------------------------------------seeting1-------------------------------
@ignore_warnings(category=ConvergenceWarning)
def LOO_decode(row, BOLD_signals, labels, cv, chance = False):
    features = BOLD_signals[:, row]
    try:
        if chance:
            #applying VarianceThreshold method (alterative methods:PCA) to extract features: removing low variance voxels
            pipeline = make_pipeline(VarianceThreshold(),
                                     StandardScaler(), # normalize: mean=0, variance=1
                                     DummyClassifier(strategy='uniform',
                                                     random_state=12345,
                                                     ))
        else:
            svm = LinearSVC(penalty='l1', ##
                            dual=False,
                            class_weight="balanced",
                            random_state=12345,
                            )    
            svm = CalibratedClassifierCV(svm, method='sigmoid',
                                         cv=5,)
            pipeline = make_pipeline(VarianceThreshold(), 
                                     StandardScaler(),
                                     svm,
                                     )
        res = cross_validate(pipeline,
                             features,
                             labels,
                             scoring='roc_auc',
                             cv=cv,
                             n_jobs=1,
                             verbose=0,
                             return_estimator=True,
                             )
        estimators = res['estimator']
        idxs_test = [idx_test for _, idx_test in cv.split(features, labels,)]
        scores = [roc_auc_score(labels[idx_test], est.predict_proba(features[idx_test])[:,-1]) for idx_test,est in zip(idxs_test,estimators)]
    except:
        scores = [0.5 for train,test in cv.split(features, labels)]
        print('broken')
    return np.array(scores)

#------------------------------define function 2-------------------------------
class MyStruct:
    pass
common_settings = MyStruct()
common_settings.radius = 6 #in mm ######### change*
common_settings.model_name = 'mvpa_rule_KFolds' ######### change*
common_settings.label_map = {0: 0, 1: 1} ######### change*
common_settings.concepts_map = {0: 'rule0', 1: 'rule1'} ######### change*

#==============================================================================
# train and test
if __name__ == "__main__":
    [os.remove(item) for item in glob('../*/core*')]
    radius = common_settings.radius
    condition = 'rulelist' # according to experiment conditions to change ################# change*
    feature_type = common_settings.model_name
    sub = 1101 #===change sub===
    sub = f'sub{sub:03d}'
    os.getcwd()
    working_dir = f'/home/xuchuanyong/Documents/DATA_analysis/MRI_SZU_HR/Decoding_mvpa/data_rulelist/{sub}' ################## change*
    mask_dir = '/home/xuchuanyong/Documents/DATA_analysis/MRI_SZU_HR/Decoding_mvpa/brainmask/'          ################## change*
    results_dir = '/home/xuchuanyong/Documents/DATA_analysis/MRI_SZU_HR/Decoding_mvpa/results/KFolds_MNI_space_rulelist' ################## change*
#   ---------------
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    working_data = os.path.join(working_dir,
                                'whole_brain_average.nii.gz')
    event_file = os.path.join(working_dir,
                              f'events_{condition}.csv')
    wholebrain_mask = os.path.join(mask_dir, 'BrainMask_05_91x109x91.nii')    
    # extract the sphere voxels
    print('extracting sphere voxels ...')
    if os.path.exists(os.path.join(working_dir, 'rows.npy')):
        rows = np.load(os.path.join(working_dir, 'rows.npy'), allow_pickle=True)
    else:
        (X, A, process_mask, process_mask_affine, process_mask_coords) = get_sphere_data(load_img(wholebrain_mask),
                                                                                         load_img(working_data), 
                                                                                         radius,
                                                                                         )
        rows = A.rows
        np.save(os.path.join(working_dir,'rows.npy'), rows, allow_pickle=True)
        del X,A,process_mask,process_mask_affine,process_mask_coords
    masker = NiftiMasker(wholebrain_mask,).fit()
    BOLD_signals = masker.transform(working_data)
    #define cross validation procedure
    df_data = pd.read_csv(event_file)
    labels = df_data[condition].map(common_settings.label_map).values # label linked to values################## change*
    #words = df_data[condition].map(common_settings.concepts_map).values # characters of label################## change* 
    cv = StratifiedShuffleSplit(n_splits = 10,test_size = .2,random_state = 12345) #times of repeat
    # word2label = {}
    # for l, w in zip(labels, words):
    #     word2label[w] = l
    # words_class0 = [key for key, val in word2label.items() if val == 0]
    # words_class1 = [key for key, val in word2label.items() if val == 1]
    # idxs_train, idxs_test = [], []
    # for word_class0 in words_class0:
    #     for word_class1 in words_class1:
    #         idxs_train.append(np.where(np.logical_and(words != word_class0,
    #                                                   words != word_class1,))[0])
    #         idxs_test.append(np.where(np.logical_or(words == word_class0,
    #                                                 words == word_class1,))[0])
    # cv = check_cv(zip(idxs_train, idxs_test),)
    row_groups = np.array_split(rows, 20) ## parcellate the brain data for faster processing
    
    for ii, row_group in enumerate(row_groups):
        score_file_name = os.path.join(results_dir,
                                       f'{sub}_{condition}_{radius}mm_{ii+1}_score.npy')
        chance_file_name = score_file_name.replace('score','chance')
        if not os.path.exists(score_file_name):
            print(f'start {ii+1}')
            # decoding
            gc.collect()
            scores = Parallel(n_jobs = -1, verbose = 1)(delayed(LOO_decode)(*[
                row, BOLD_signals, labels, cv, False]) for row in row_group)
            gc.collect()
            # chance level decoding
            chances = Parallel(n_jobs = -1, verbose =1)(delayed(LOO_decode)(*[
                row, BOLD_signals, labels, cv, True]) for row in row_group)
            gc.collect()
            
            np.save(score_file_name, np.array(scores))
            np.save(chance_file_name, np.array(chances))
        else:
            print(f'{score_file_name} exits')


#==============================================================================
# indivudual level: one sample t-test: test - chance, to build a p-value/t-value nii map
scores_list = np.sort(glob(os.path.join(results_dir, f'{sub}_{condition}_{radius}mm_*_score.npy')))
scores = np.concatenate([np.load(score, allow_pickle=True) 
                         for score in scores_list ], axis=0)
chances_list = np.sort(glob(os.path.join(results_dir, f'{sub}_{condition}_{radius}mm_*_chance.npy')))
chances = np.concatenate([np.load(chance, allow_pickle=True) 
                          for chance in chances_list ], axis=0)

scores_chances = scores - chances
scores_chances_brain = np.mean(scores_chances, axis=1)
scores_chances_brain = masker.inverse_transform(scores_chances_brain.reshape(1,-1))
scores_chances_brain_name = os.path.join(results_dir,
                                           f'{sub}_{condition}_{radius}mm_scores_chances_brain.nii.gz')
nib.save(scores_chances_brain, scores_chances_brain_name)

tvalues, pvalues = [], []
tvalues, pvalues = ttest_1samp(scores_chances, 0, axis=1)
tvalues_brain = masker.inverse_transform(tvalues.reshape(1, -1))
pvalues_brain = masker.inverse_transform(pvalues.reshape(1, -1))
t_file_name = os.path.join(results_dir,
                                       f'{sub}_{condition}_{radius}mm_tvalues_brain.nii.gz')
p_file_name = t_file_name.replace('tvalues_brain','pvalues_brain')
nib.save(tvalues_brain,t_file_name)#save coverted nii
nib.save(pvalues_brain,p_file_name)#save coverted nii

#==============================================================================
# group level: FSL randomise for one sample t-test
