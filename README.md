Thanks for Ning Mei providing these python script templates for decoding
# For reading papaers to understand RSA
## The questions for variant distance measurments
<br>1. Walther, A., Nili, H., Ejaz, N., Alink, A., Kriegeskorte, N., and Diedrichsen, J. (2016). Reliability of dissimilarity measures for multi-voxel pattern analysis. NeuroImage 137, 188–200. https://doi.org/10.1016/j.neuroimage.2015.12.012.
<img width="587" height="726" alt="image" src="https://github.com/user-attachments/assets/139971e8-6d6e-416f-865a-3627978c0c93" />

<br>该文聚焦在磁共振数据的表征相似性分析，系统分析前人研究数据并结合simulation发现:
<br>(1)为了提升表征不相似性距离RDM构建的可靠性，任何距离度量都应该采用*多变量噪声标准化(multivariate noise normalization)*;
<br>(2)连续距离度量优于离散距离度量（如LDA、SVM等），能提升稳定性、信息量；
<br>(3)欧式、马氏、相关距离的可靠性相似，但相关距离的可解释性稍差；
<br>(4)交叉验证距离估计是无偏的(unbias)，具备有意义的基线零点，使得距离比值可解释。
总结：坐着推荐采用交叉验证的马氏距离；即使不计算马氏距离，但至少应结合交叉验证、多变量噪声标准化，来构建距离矩阵。


## Toolbox
<br> rsatoolbox (https://rsatoolbox.readthedocs.io/en/stable/index.html; https://github.com/rsagroup/rsatoolbox/tree/main)



# For reading papaers to understand MVPA
