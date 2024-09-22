#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:46:05 2023

@author: nmei
"""
import os,common_settings,gc
import numpy as np
import pandas as pd
from glob import glob
from nilearn.maskers import NiftiMasker
from nilearn.image import load_img
from utils import get_sphere_data,concept_mapping
from sklearn.model_selection import check_cv,cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
try:
    from sklearn.utils.testing import ignore_warnings
    from sklearn.exceptions import ConvergenceWarning
except:
    from sklearn.utils._testing import ignore_warnings
    from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel,delayed

@ignore_warnings(category=ConvergenceWarning)
def LOO_decode(row,BOLD_signals,labels,words,cv,chance = False):
    features = BOLD_signals[:,row]
    try:
        if chance:
            pipeline = make_pipeline(VarianceThreshold(),
                                     StandardScaler(),
                                     DummyClassifier(strategy = 'uniform',
                                                     random_state = 12345,
                                                     ))
        else:
            svm = LinearSVC(penalty = 'l1',
                            dual = False,
                            class_weight = "balanced",
                            random_state = 12345,
                            )
            svm = CalibratedClassifierCV(svm,method = 'sigmoid',
                                         cv = 5,)
            pipeline = make_pipeline(VarianceThreshold(),
                                     StandardScaler(),
                                     svm,
                                     )
        res = cross_validate(pipeline,
                             features,
                             labels,
                             groups = words,
                             scoring = 'roc_auc',
                             cv = cv,
                             n_jobs = 1,
                             verbose = 0,
                             return_estimator = True,
                             )
        estimators = res['estimator']
        idxs_test = [idx_test for _,idx_test in cv.split(features,labels,)]
        scores = [roc_auc_score(labels[idx_test], est.predict_proba(features[idx_test])[:,-1]) for idx_test,est in zip(idxs_test,estimators)]
    except:
        scores = [0.5 for train,test in cv.split(features,labels,groups = words)]
    
    return np.array(scores)

if __name__ == "__main__":
    [os.remove(item) for item in glob('../*/core*')]
    radius = common_settings.radius
    condition = 'likableness' # change condition
    feature_type = common_settings.model_name
    sub = 11 # change sub
    sub = f'sub-{sub:03d}'
    
    working_dir = f'../data/native/{sub}'
    results_dir = '../results/LOO_native_space'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    # define working data
    working_data = os.path.join(working_dir,
                                'whole_brain.nii.gz')
    wholebrain_mask = os.path.join(working_dir,
                                   'whole_brain_mask.nii.gz')
    event_file = os.path.join(working_dir,
                              f'events_{condition}.csv')
    # extract the sphere voxels
    print('extracting ...')
    (X,A,
     process_mask,
     process_mask_affine,
     process_mask_coords)= get_sphere_data(load_img(wholebrain_mask),
                                           load_img(working_data),
                                           radius,
                                           )
    
    masker = NiftiMasker(wholebrain_mask,).fit()
    BOLD_signals = masker.transform(working_data)
    # define cross validation procedure
    df_data = pd.read_csv(event_file)
    labels = df_data['targets'].map(common_settings.label_map).values
    words = df_data['concepts'].map(concept_mapping()).values
    word2label = {}
    for l,w in zip(labels,words):
        word2label[w] = l
    words_class0 = [key for key,val in word2label.items() if val == 0]
    words_class1 = [key for key,val in word2label.items() if val == 1]
    idxs_train,idxs_test = [],[]
    for word_class0 in words_class0:
        for word_class1 in words_class1:
            idxs_train.append(np.where(np.logical_and(words != word_class0,
                                             words != word_class1,))[0])
            idxs_test.append(np.where(np.logical_or(words == word_class0,
                                           words == word_class1,))[0])
    cv = check_cv(zip(idxs_train,idxs_test),)
    rows = A.rows
    row_groups = np.array_split(rows,20)
    for ii,row_group in enumerate(row_groups):
        score_file_name = os.path.join(results_dir,
                                       f'{sub}_{condition}_{radius}mm_{ii+1}_score.npy')
        chance_file_name = score_file_name.replace('score', 'chance')
        if not os.path.exists(score_file_name):
            print(f'start {ii+1}')
            # decoding
            gc.collect()
            scores = Parallel(n_jobs = -1,verbose = 1)(delayed(LOO_decode)(*[
                row,BOLD_signals,labels,words,cv,False]) for row in row_group)
            gc.collect()
            # chance level decoding
            chances = Parallel(n_jobs = -1,verbose = 1)(delayed(LOO_decode)(*[
                row,BOLD_signals,labels,words,cv,True]) for row in row_group)
            gc.collect()
            
            np.save(score_file_name,np.array(scores))
            np.save(chance_file_name,np.array(chances))
        else:
            print(f'{score_file_name} exits')
