%% script to run 50 indepedent datasplits for ZSL
clear;

addpath('../function');

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
lambda = 1e-4;

%% Do M independent splits
M = 5;
meanAcc = zeros(1,M);

for trial = 1:M
    
    func_tr_ZSL_Attr_zscore(perc_TrainingSet,cluster_type,nSample,CodebookSize,process,...
        FEATURETYPE,nPCA,C,trial,EmbeddingMethod,lambda);
    [meanAcc(trial),Acc{trial}] = func_ts_ZSL_Attr_zscore(perc_TrainingSet,cluster_type,nSample,CodebookSize,...
        process,FEATURETYPE,nPCA,C,SelfTraining,trial,EmbeddingMethod,lambda);

    fprintf('%dth trial acc = %.2f\n',trial,meanAcc(trial)*100);
%     fprintf('mean Acc=%.2f\n',mean(meanAcc)*100);
end

mean(meanAcc)
std(meanAcc)

%% Save Result
save_path = '/import/geb-experiments-archive/Alex/UCF101/ITF/FV/Zeroshot/jointcodebook/Embedding/DatasetSplit_tr-0.5_ts-0.5/HOF|HOG|MBH/';
save(sprintf([save_path,'meanAcc_50_lambda_%g_norm-zscore.mat'],lambda),'meanAcc','Acc');