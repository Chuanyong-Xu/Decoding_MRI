##notes:* load the specific node can using anaconda
ssh -X fox-c04

##notes:*
module load anaconda3

##notes:* creat new environment
conda env list
#conda create -n decoding_rsa python=3.10

##notes:* 
source activate decoding_rsa
conda activate decoding_rsa
#conda install numpy
#conda install spyder
spyder &

#conda deactivate decoding_rsa


##notes: *backup* some packages are not convenient to install
 # conda install -c conda-forge *package name*
 #or
 #pip install *package name*

#find /usr -type d -name "qt5"
#export QT_PLUGIN_PATH=/usr/lib64/qt5/plugins
#pip install --upgrade spyder
#pip install --upgrade numpy scipy pyqt5
