%% Regularized Multi-Task Learning to train regression
%
%   each dimension in word vector is treated as a single task

function ReturnVar=func_tr_ZSL_GOMTL_L2_ncg(Para)

global D idx_HMDB51 idx_UCF101 idx_OlympicSports idx_CCV tr_LabelVec_HMDB51 tr_LabelVec_UCF101 tr_LabelVec_OlympicSports tr_LabelVec_CCV;

addpath('/import/geb-experiments/Alex/CVPR15/Code/Zeroshot/HMDBregression/Code/Basic/FVNormalization');
addpath('/import/geb-experiments/Alex/ICCV15/code/TransferRegression/CollectData/function');
addpath('/import/geb-experiments/Alex/ICCV15/code/SharedTools/function/');
addpath('/import/geb-experiments/Alex/Matlab/Techniques/poblano_toolbox');

%% Generate Training Kernel Matrix
selected_tr_idx = Para.tr_sample_ind'.*idx_UCF101;
selected_tr_idx = selected_tr_idx(selected_tr_idx~=0);
aug_tr_idx = [idx_HMDB51 selected_tr_idx idx_OlympicSports idx_CCV];

Z = [tr_LabelVec_HMDB51 ; tr_LabelVec_UCF101(Para.tr_sample_ind,:) ; tr_LabelVec_OlympicSports ; tr_LabelVec_CCV];

Z = func_L2Normalization(Z)';
Z_sumt = sum(Z,1);
T = size(Z,1); % number of tasks
N = size(Z,2); % number of instances

K = D(aug_tr_idx,aug_tr_idx);

%% Iterative Optimize A and S
N_L = round(size(Z,1)*Para.LatentProportion);
[ReturnVar.S , ReturnVar.A , ReturnVar.L]=func_KernelizedGOMTL_ncg_admm(K,Z,N_L,Para.lambdaS,Para.lambdaA);




