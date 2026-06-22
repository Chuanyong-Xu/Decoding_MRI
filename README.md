Thanks for Ning Mei providing these python script templates for decoding

# For reading papaers to understand RSA
## The questions for variant distance measurments
<img width="926" height="329" alt="image" src="https://github.com/user-attachments/assets/0267da2e-1422-468c-b110-49d099655c28" />

<br>1. Walther, A., Nili, H., Ejaz, N., Alink, A., Kriegeskorte, N., and Diedrichsen, J. (2016). Reliability of dissimilarity measures for multi-voxel pattern analysis. NeuroImage 137, 188–200. https://doi.org/10.1016/j.neuroimage.2015.12.012.
<img width="587" height="726" alt="image" src="https://github.com/user-attachments/assets/139971e8-6d6e-416f-865a-3627978c0c93" />

<br>该文聚焦在磁共振数据的表征相似性分析，系统分析前人研究数据并结合simulation发现:
<br>(1)为了提升表征不相似性距离RDM构建的可靠性，任何距离度量都应该采用*多变量噪声标准化(multivariate noise normalization)*;
<br>(2)连续距离度量优于离散距离度量（如LDA、SVM等），能提升稳定性、信息量；
<br>(3)欧式、马氏、相关距离的可靠性相似，但相关距离的可解释性稍差；
<br>(4)交叉验证距离估计是无偏的(unbias)，具备有意义的基线零点，使得距离比值可解释。
<br>交叉验证马氏距离(crossnobis)计算过程：马氏距离的核心思想是：在计算两个条件表征模式之间的距离时，不仅考虑模式均值差异，还考虑各特征维度的噪声方差及特征间的噪声协方差。
#### 总结：作者推荐采用交叉验证的马氏距离；即使不计算马氏距离，但至少应结合交叉验证、多变量噪声标准化，来构建距离矩阵。

#### 一、fMRI 数据中的计算流程：
fMRI 数据通常先通过一阶 GLM 得到每个条件的 response pattern。对于一个被试、一个 ROI 或一个 searchlight，数据结构可以表示为：
（1）原始时序数据：time points × voxels，即每个体素在每个时间点上的 BOLD 信号；
（2）GLM 后的条件激活模式：conditions × voxels × runs，其中 conditions 是实验条件数，voxels 是 ROI 或 searchlight 中的体素数，runs 是扫描 run 数；
（3）GLM residuals：time points × voxels × runs，用于估计每个体素以及体素之间的噪声结构(注意是：time points)；
（4）最终 RDM：conditions × conditions，或展开为 condition pairs × 1 的向量。

<br>具体计算时，首先在每个 run 内分别估计每个条件的 beta pattern。每个 beta pattern 是一个向量，长度等于 ROI 或 searchlight 中的体素数。例如，如果有 9 个条件和 120 个体素，那么每个 run 会得到一个 9 × 120 的条件模式矩阵。
<br>然后，利用 GLM residuals 估计噪声协方差矩阵。这个矩阵描述的是不同体素噪声之间的相关结构。普通欧氏距离默认每个体素的噪声方差相同、体素之间噪声相互独立；而马氏距离会根据噪声协方差对体素维度进行重新加权。噪声大的体素会被降低权重，噪声高度相关的体素也不会被当作完全独立的信息来源。
<br>接着，对每个条件的 beta pattern 进行多变量噪声标准化，也就是 whitening。whitening 后的 pattern 可以理解为已经去除了不同体素噪声方差和噪声相关性的影响。在这个标准化空间中计算两个条件 pattern 的 squared Euclidean distance，就等价于在原始空间中计算 squared Mahalanobis distance。

<img width="595" height="102" alt="image" src="https://github.com/user-attachments/assets/85411cdb-a9b6-4c71-aae1-2c1cc4fcaf95" />

<br>如果计算普通马氏距离，可以先把所有 runs 中同一个条件的 beta pattern 平均，得到一个 conditions × voxels 的矩阵，然后在多变量噪声标准化后的 pattern 上计算所有条件两两之间的距离，最终得到一个 conditions × conditions 的 RDM。
但是，普通马氏距离仍然会受到噪声正偏差的影响。即使两个条件真实上没有差异，只要 beta pattern 中有噪声，估计出来的距离也往往大于 0。因此，更推荐使用交叉验证马氏距离，也就是 crossnobis / LDC。
<br>crossnobis 的核心是把数据分成相互独立的 partitions，fMRI 中通常使用 run 作为 partition。以 leave-one-run-out 为例，每次取一个 run 作为测试 partition，剩余 runs 作为训练 partition。对于任意两个条件，分别在测试 run 和训练 runs 中计算这两个条件的 pattern difference。然后，用训练数据或对应 residuals 估计噪声协方差，并用该噪声结构对 pattern difference 进行加权。最后，将测试 run 中的 condition difference 与训练 runs 中的 condition difference 进行交叉乘积，得到该条件对的 crossnobis distance。
这个过程对所有 runs 重复一次，并把所有 folds 的结果平均。最终得到每个条件对的交叉验证马氏距离。对于 n 个条件，最终会得到一个 n × n 的 RDM，或者取上/下三角。
<br>需要注意的是，crossnobis 是 squared distance 的交叉验证估计，不需要开方。它可以出现负值，负值不是错误，而是无偏估计的正常结果。如果两个条件真实上没有系统性差异，crossnobis 的期望值为 0，但在有限样本中可以围绕 0 正负波动。因此，不应该把负值强行设为 0。

#### 二、针对 EEG 数据的计算过程
<br>EEG 数据没有 fMRI 中天然的 voxel pattern 和 GLM beta pattern，但可以采用完全类似的逻辑，把电极或电极-时间窗作为 multivariate features。对于一个被试，数据结构通常可以表示为：
<br>（1）原始分段 EEG 数据：trials × channels × time points，其中 trials 是试次数，channels 是电极数，time points 是反馈锁定或刺激锁定后的时间点；
<br>（2）条件标签：trials × 1，例如 9 个条件可以由 errors × difficulty 构成；
<br>（3）run/block/fold 标签：trials × 1，用于定义交叉验证 partition；
<br>（4）某一时间点或时间窗的数据矩阵：trials × features，其中 features 可以是 channels，也可以是 channels × time-window samples；
<br>（5）最终 RDM：subjects × time points × condition pairs，例如 9 个条件时，condition pairs 的数量为 36。

<br>具体计算时，首先在每个时间点或滑动时间窗内提取 EEG pattern。如果只使用当前时间点，则每个 trial 的 pattern 是一个 1 × channels 的向量；如果使用一个时间窗，则可以先对时间窗内信号平均，得到 1 × channels 的向量，也可以把时间窗内所有采样点拼接起来，得到 1 × (channels × window samples) 的向量。前一种方式更稳定，后一种方式包含更多时间信息但维度更高，对 trial 数要求更高。
<br>然后，根据实验条件把 trials 分成多个 conditions。例如，在 errors 有 3 个水平、difficulty 有 3 个水平时，可以构成 9 个条件。对于每个 condition 和每个 fold，计算该 condition 在该 fold 内的平均 EEG pattern。这样，在每个时间点会得到一个 conditions × folds × features 的矩阵。
<br>接着，需要估计 EEG 特征维度上的噪声结构。具体做法是，在训练 folds 中，先减去每个 condition 的平均 pattern，得到 trial-level residuals。这个 residual 矩阵的结构是 training trials × features。然后用这些 residuals 估计 features 之间的噪声协方差。这里的 features 可以是电极，也可以是电极与时间窗采样点的组合。
<br>对于 EEG 数据，是否使用完整的多变量协方差需要谨慎。如果 features 数量较多、每个 condition 的 trial 数较少，完整协方差矩阵可能不稳定。因此更稳妥的选择通常是使用 shrinkage covariance，或者先使用 diagonal covariance。diagonal covariance 只校正每个电极或特征自身的噪声方差，不估计特征之间的噪声相关；shrinkage covariance 则在完整协方差和对角协方差之间折中，既考虑特征之间的相关性，又避免协方差矩阵过度不稳定。
<br>在得到噪声协方差后，对每个 condition pattern 进行多变量噪声标准化。对于 EEG 来说，这一步可以理解为：噪声大的电极或特征被降低权重，噪声相关较强的电极组合不会被重复计算为独立信息。因此，标准化后的 EEG pattern 更接近于按信噪比加权后的多变量表征。
如果计算普通马氏距离，可以把所有 trials 或所有 folds 中同一个 condition 的 pattern 平均，得到一个 conditions × features 的矩阵，然后在噪声标准化后的 feature 空间中计算两两 condition 之间的距离，得到 conditions × conditions 的 RDM。
如果计算 crossnobis，则需要使用独立 folds。最理想的 fold 是真实 run 或 block；如果没有真实 run/block（或者每个run之间的条件数很不平衡），也可以在每个 condition 内进行平衡随机划分，但这种做法不如真实 run/block 严格。以 leave-one-fold-out 为例，每次取一个 fold 作为测试 partition，剩余 folds 作为训练 partition。对于每一对 conditions，分别在测试 fold 和训练 folds 中计算它们的 EEG pattern difference。然后使用训练 folds 的 residuals 估计噪声结构，并用该噪声结构对 pattern difference 进行加权。最后，将测试 fold 中的 condition difference 与训练 folds 中的 condition difference 进行交叉乘积，得到该条件对在当前 fold 下的 crossnobis distance。
<br>这一过程对所有 folds 重复，并对 folds 取平均。最终，在每个时间点都会得到一个 conditions × conditions 的 EEG RDM。如果有 9 个条件，则每个时间点得到 36 个 pairwise distances；如果有多个被试，最终数据结构就是 subjects × time points × 36。
<br>需要特别注意的是，EEG 的 crossnobis 结果可能比 correlation distance 更不平滑。这是因为 crossnobis 不再是被限制在固定范围内的相关距离，而是一个未归一化的、可以为负的连续距离估计。它对 fold 内 trial 数、feature 数量、噪声协方差估计、以及是否使用真实 run/block 作为 partition 都比较敏感。因此，在 EEG 数据中计算 crossnobis 时，通常需要控制 feature 维度、平衡每个 condition 的 trial 数，并优先使用真实 run/block 作为交叉验证 folds。


<br>Q: 个人前期经验，当feature远大于样本数时，噪声协方差估计是不稳定的，此时或许不一定完全优于一般的相关距离计算？
<br>但是自行手动写代码计算或者使用AI辅助均来实现crossnobis显得比较麻烦，可以直接采用目前成熟的系列工具包，此处列出的是RSA方法系统提出者Kriegeskorte团队开发的工具包。
## Toolbox
<br> rsatoolbox (https://rsatoolbox.readthedocs.io/en/stable/index.html; https://github.com/rsagroup/rsatoolbox/tree/main)



# For reading papaers to understand MVPA
