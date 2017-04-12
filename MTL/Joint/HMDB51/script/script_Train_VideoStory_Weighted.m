%% script to compute weighted VideoStory


%% Load Data Split
datasplit_path = sprintf('/import/vision-datasets2/HMDB51/hmdb51_%s_ZeroRegression/DatasetSplit/',Para.process);
load(sprintf([datasplit_path,'DatasetSplit_tr-%.1f_ts-%.1f_t-%d.mat'],Para.perc_TrainingSet,Para.perc_TestingSet,trial));
Para.idx_TrainingSet = sort(idx_TrainingSet,'ascend');
Para.idx_TestingSet = sort(idx_TestingSet,'ascend');
clear idx_TrainingSet idx_TestingSet;

%% Prepare Training Data
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

%% Load weight
XYmodel_filepath = sprintf([weight_model_path,'LinearRegress_trial-%d_embed-%s_sigmaX-%g_sigmaY-%g_aug-1_norm-zscore_ccv.mat'],trial,Para.EmbeddingMethod,Para.sigmaX,Para.sigmaY);
KLIEP_XY = load(XYmodel_filepath,'Model');
weight = KLIEP_XY.Model.weight;
%% Load Weight
%                     XYmodel_filepath = sprintf([weight_model_path,'LinearRegress_trial-%d_embed-%s_sigmaX-%g_sigmaY-%g_aug-1.mat'],trial,Para.EmbeddingMethod,Para.sigmaX,Para.sigmaY);
%                     KLIEP_XY = load(XYmodel_filepath,'Model');
%                     weight = KLIEP_XY.Model.weight;

%% Generate Training Kernel Matrix
selected_tr_idx = Para.tr_sample_ind'.*idx_HMDB51;
selected_tr_idx = selected_tr_idx(selected_tr_idx~=0);
aug_tr_idx = [ selected_tr_idx idx_UCF101 idx_OlympicSports idx_CCV];

Z_All = zscore(func_L2Normalization([tr_LabelVec_HMDB51 ; tr_LabelVec_UCF101 ; tr_LabelVec_OlympicSports ; tr_LabelVec_CCV]));
Z = Z_All(aug_tr_idx,:)';


N_l = numel(weight);
K = D(aug_tr_idx,aug_tr_idx);

%% Train GOMTL Model
param.MaxItr = 30;
param.epsilon = 5e-3;
[Model.D, Model.A , Model.S , Model.L]=func_KernelizedVideoStory_Weighted(K,Z,LatentDim,weight,lambdaD,lambdaS,lambdaA,param);

%%% Save Result
save(model_filepath,'Model');

