#concact
  work_path=/home/xuchuanyong/Documents/DATA_analysis/MRI_SZU_HR/Decoding_mvpa/results/KFolds_MNI_space_rule_respons/
  filename=design
  cd $work_path
  #creat .con, one-side positive t
#/RequiredEffect		1.102
  cat <<EOL > $filename.con
/ContrastName1 Positive_Test
/NumWaves	1
/NumContrasts	1
/PPheights	1.000000e+00

/Matrix
1
EOL



