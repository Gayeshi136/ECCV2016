%% function to load datasplit

function [idx_TrainingSet,idx_TestingSet]=func_LoadDataSplit(Para)


%% Load Data Split
load(sprintf([Para.zeroshot_datasplit_path,'DatasetSplit_tr-%.1f_ts-%.1f_t-%d.mat'],...
    Para.perc_TrainingSet,Para.perc_TestingSet,Para.trial));
idx_TrainingSet = sort(idx_TrainingSet,'ascend');
idx_TestingSet = sort(idx_TestingSet,'ascend');
