#concact
  work_path=/home/xuchuanyong/Documents/DATA_analysis/MRI_SZU_HR/Decoding_mvpa/results/smooth6_KFolds_MNI_space_rule_respons/ #KFolds_MNI_space_rule_respons/
  filename=6mm_scores_chances_brain
  cd $work_path
  fslmerge -t $filename"_all" *$filename.nii.gz
  fslmaths 6mm_scores_chances_brain_all.nii.gz -Tmean 6mm_scores_chances_brain_all_mean
  fslmaths 6mm_scores_chances_brain_all_mean.nii.gz -thr 0 -bin 6mm_scores_chances_brain_all_mean_mask
  #randomise -i $filename"_all" -o $filename"_all_onesampT" -1 -T
  randomise_parallel -i $filename"_all".nii.gz -o all_onesampT -d design.mat -t design.con -m 6mm_scores_chances_brain_all_mean_mask -n 5000 -T
  #randomise_parallel -i $filename"_all" -o $filename"_all_onesampT" -1 -v 5 -T



