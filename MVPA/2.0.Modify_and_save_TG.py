#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 18:57:22 2023

@author: MeiNing_01
"""

import os
from shutil import copyfile,rmtree
copyfile('../experiment_settings.py','experiment_settings.py')
import experiment_settings as ES


template = '2.3.Decoding_2_mvpa_Dev_lev.py'
scripts_folder = 'Dev_lev_decoding_scripts'
if not os.path.exists(scripts_folder):
    os.mkdir(scripts_folder)
else:
    rmtree(scripts_folder)
    os.mkdir(scripts_folder)
os.mkdir(f'{scripts_folder}/outputs')
#copyfile('../utils.py',f'{scripts_folder}/utils.py')
copyfile('../experiment_settings.py',f'{scripts_folder}/experiment_settings.py')

collections = []
for sub in list(ES.subject_info.keys()):
    new_script_name = os.path.join(scripts_folder,f'tempo_gen_{sub}.py')
    with open(new_script_name,'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "#===change sub===" in line:
                    line = f"    sub = {sub}\n"
                new_file.write(line)
            old_file.close()
        new_file.close()
    
    new_bash_script = os.path.join(scripts_folder,f'decode_{sub}')
    content = f'''#!/bin/bash
# SBATCH --job-name=D-{sub}
# SBATCH --output=outputs/out_{sub}.txt
# SBATCH --error=outputs/err_{sub}.txt
# SBATCH --time=24:00:00
# SBATCH --ntasks-per-node=-1
# SBATCH --cpus-per-task={ES.n_jobs}
# SBATCH --nodes={ES.node}
# SBATCH --mem-per-cpu={ES.mem}G

module load anaconda3
source activate decoding_rsa
cd $SLURM_SUBMIT_DIR

python3 tempo_gen_{sub}.py
'''
    with open(new_bash_script,'w') as f:
        f.write(content)
        f.close()
    collections.append(f'decode_{sub}')

