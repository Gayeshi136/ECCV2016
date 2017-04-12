%% script to compute KLIEP XY weight

perc_TrainingSet = 0.5;
perc_TestingSet = 1 - perc_TrainingSet;
cluster_type = 'vlfeat';
nSample = 256000;
CodebookSize = 128;
process = 'org'; % preprocess of dataset: org,sta
FEATURETYPE = 'HOF|HOG|MBH';
nPCA = 0;
C = 2^1; % Cost parameter for SVR
SelfTraining = 0;   % Indicator if do selftraining
trial = 1;
EmbeddingMethod = 'add';

%% Load Data Split
datasplit_path = sprintf('/import/vision-datasets2/HMDB51/hmdb51_%s_ZeroRegression/DatasetSplit/',process);
load(sprintf([datasplit_path,'DatasetSplit_tr-%.1f_ts-%.1f_t-%d.mat'],perc_TrainingSet,perc_TestingSet,trial));
Para.idx_TrainingSet = sort(idx_TrainingSet,'ascend');
Para.idx_TestingSet = sort(idx_TestingSet,'ascend');
clear idx_TrainingSet idx_TestingSet;

%% Prepare Training Data
tr_LabelVec = [];
tr_sample_ind = zeros(size(Para.sample_class_ind,1),1);   % train sample index

for c_tr = 1:length(Para.idx_TrainingSet)
    
    %% Extract Training Features for each class
    currentClassName = Para.ClassLabelsPhrase{Para.idx_TrainingSet(c_tr)};
    tr_sample_class_ind = ismember(Para.sample_class_ind(:,1),currentClassName);
    tr_sample_ind = tr_sample_ind + tr_sample_class_ind;
end

Para.tr_sample_ind = logical(tr_sample_ind);
Para.ts_sample_ind = ~tr_sample_ind;
clear tr_sample_ind ts_sample_ind;

%% X kernel matrix
%%% Generate Training Instance to Testing Instance Kernel
selected_ts_idx = Para.ts_sample_ind'.*idx_HMDB51;
Para.selected_ts_idx = selected_ts_idx(selected_ts_idx~=0);
L2Dist = 2*D(1)-2*D(Para.selected_ts_idx,Para.selected_ts_idx);
Kernel_ts2ts_X = exp(-L2Dist/Para.sigmaX^2);

%%% Generate Training Kernel Matrix
selected_tr_idx = Para.tr_sample_ind'.*idx_HMDB51;
selected_tr_idx = selected_tr_idx(selected_tr_idx~=0);
aug_tr_idx = [ selected_tr_idx idx_UCF101 idx_OlympicSports idx_CCV];

L2Dist = 2*D(1)-2*D(Para.selected_ts_idx,aug_tr_idx);
Kernel_ts2tr_X = exp(-L2Dist/Para.sigmaX^2);

%% Y Kernel Matrix
%%% Generate Training Instance to Testing Instance Kernel
selected_ts_idx = Para.ts_sample_ind'.*idx_HMDB51;
Para.selected_ts_idx = selected_ts_idx(selected_ts_idx~=0);

%%% Generate Training Kernel Matrix
selected_tr_idx = Para.tr_sample_ind'.*idx_HMDB51;
selected_tr_idx = selected_tr_idx(selected_tr_idx~=0);
aug_tr_idx = [ selected_tr_idx idx_UCF101 idx_OlympicSports idx_CCV];

[Z_All,mu,sig]= zscore(func_L2Normalization([tr_LabelVec_HMDB51 ; tr_LabelVec_UCF101 ; tr_LabelVec_OlympicSports ; tr_LabelVec_CCV]));
Z_tr = Z_All(aug_tr_idx,:)';
%         [C,ia,ic] = unique(Z_tr','rows','stable');

%         Z_tr = func_L2Normalization(Z_tr)';
Z_sumt = sum(Z_tr,1);
T = size(Z_tr,1); % number of tasks
N = size(Z_tr,2); % number of instances

Z_tr_proto = unique(Z_tr','rows','stable');
Z_ts_proto = (func_L2Normalization(Para.phrasevec_mat) -...
    repmat(mu,size(Para.phrasevec_mat,1),1))./repmat(sig,size(Para.phrasevec_mat,1),1);
Z_ts_proto = Z_ts_proto(Para.idx_TestingSet,:);

L2Dist = pdist2(Z_ts_proto,Z_tr','cosine');
Kernel_ts2tr_Y = exp(-L2Dist/Para.sigmaY^2);

L2Dist = pdist2(Z_ts_proto,Z_ts_proto,'cosine');
Kernel_ts2ts_Y = exp(-L2Dist/Para.sigmaY^2);

%% Learn Weight
weight = func_KLIEP_XY_Naive(Kernel_ts2ts_X,Kernel_ts2tr_X,Kernel_ts2ts_Y,Kernel_ts2tr_Y);

%% Train Weighted Model
% N_l = numel(weight);
% S = Z_tr*sparse(1:N_l,1:N_l,sqrt(weight));
% 
% %%% Solve Ridge Regression with close-form solution
% Model.A = S/(D(aug_tr_idx,aug_tr_idx)*sparse(1:N_l,1:N_l,sqrt(weight))+lambda*N_l*eye(N_l));
Model.weight = weight;
% 
% %     Model = func_tr_ZSL_GOMTL_Dim_ncg(Para);
% 
% %%% Save Result
save(XYmodel_filepath,'Model');