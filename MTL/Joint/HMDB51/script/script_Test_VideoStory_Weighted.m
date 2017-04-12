%% script to test weighted VideoStory

%% Load Data Split
datasplit_path = sprintf('/import/vision-datasets2/HMDB51/hmdb51_%s_ZeroRegression/DatasetSplit/',process);
load(sprintf([datasplit_path,'DatasetSplit_tr-%.1f_ts-%.1f_t-%d.mat'],perc_TrainingSet,perc_TestingSet,trial));
Para.idx_TrainingSet = sort(idx_TrainingSet,'ascend');
Para.idx_TestingSet = sort(idx_TestingSet,'ascend');
clear idx_TrainingSet idx_TestingSet;

%% Prepare Training Data
tr_LabelVec = [];
tr_sample_ind = zeros(size(sample_class_ind,1),1);   % train sample index

for c_tr = 1:length(Para.idx_TrainingSet)
    
    %% Extract Training Features for each class
    currentClassName = ClassLabelsPhrase{Para.idx_TrainingSet(c_tr)};
    tr_sample_class_ind = ismember(sample_class_ind(:,1),currentClassName);
    tr_sample_ind = tr_sample_ind + tr_sample_class_ind;
end

Para.tr_sample_ind = logical(tr_sample_ind);
Para.ts_sample_ind = ~tr_sample_ind;
clear tr_sample_ind ts_sample_ind;

%% Generate Testing Kernel Matrix
selected_tr_idx = Para.tr_sample_ind'.*idx_HMDB51;
selected_tr_idx = selected_tr_idx(selected_tr_idx~=0);
Para.selected_tr_idx = [ selected_tr_idx idx_UCF101 idx_OlympicSports idx_CCV];
selected_ts_idx = Para.ts_sample_ind'.*idx_HMDB51;
Para.selected_ts_idx = selected_ts_idx(selected_ts_idx~=0);

%% Test VideoStory Model
[AccLat,Acc] = func_ts_ZSL_VideoStory_ACC_zscore_Latent(Para,Model);
