%% Test Linear Regression and Word2Vec Embedding
%
%   varargin - {1} perc_TrainingSet, {2} cluster_type - the technique used for
%   Kmeans, {3} nSample - the number of samples for generating codebook, {4}
%   CodebookSize - the number of centers for Kmeans, varargin{5} - preprocess of
%   dataset: org (original video) or sta (stabilized video), varargin{6] -
%   featuretype, varargin{7} - do pca on input data, if 0 don't do pca if
%   nonzero, do pca and take the first # varargin{7} dims as the process
%   training data.
%   rbdchisq kernel is applied
%   Self-training is applied to drift prototypes
%

function [meanAcc,Acc] = func_ts_ZSL_VideoStory_ACC_zscore_Latent_PP(Para,Model)

global D;

addpath('/import/geb-experiments/Alex/CVPR15/Code/Zeroshot/HMDBregression/Code/Basic/FVNormalization');
addpath('/import/geb-experiments/Alex/ICCV15/code/TransferRegression/CollectData/function');
addpath('/import/geb-experiments/Alex/ICCV15/code/SharedTools/function/');


%% Generate Testing Kernel Matrix


aug_tr_idx = Para.selected_tr_idx;
selected_ts_idx = Para.selected_ts_idx;

%% Testing Ridge Regression model
D_map = Model.D;
A = Model.A;
S = Model.S;
L = Model.L;

S_ts = A*D(aug_tr_idx,selected_ts_idx);   %D(selected_ts_idx,aug_tr_idx) * A;
% S_ts = func_L2Normalization(S_ts)';
S_ts = zscore(S_ts');

%% Knn to predict final labels
ts_ClassNo = Para.ClassNoPerVideo(Para.ts_sample_ind);

%%% Generate Prototypes
[Prototype,mu,sigma] = zscore(func_L2Normalization(Para.phrasevec_mat));
Prototype = (Prototype(Para.idx_TestingSet,:)*pinv(D_map)');

parfor K = 80:150
    
    stPrototype = func_SelfTraining(Prototype, S_ts, K);
    
    D_rank = pdist2(stPrototype,S_ts,'cosine');
    D_rank(isnan(D_rank)) = mean(D_rank(~isnan(D_rank)));
    D_rank = D_rank./repmat(sqrt(sum(D_rank.^2,2)),1,size(D_rank,2));
    
    [~,predict_ClassNo] = min(D_rank,[],1);
    
    %%% Calculate Average Precision for each Class
    Accuracy = zeros(1,length(Para.idx_TestingSet));
    for c_ts = 1:length(Para.idx_TestingSet)
        
        currentClass = Para.idx_TestingSet(c_ts);
        currentClass_SampleIndex = ts_ClassNo==currentClass;
        currentClass_Predict = predict_ClassNo(currentClass_SampleIndex);   % predicted class no
        Accuracy(c_ts) = sum(currentClass_Predict == c_ts)/length(currentClass_Predict);
        
    end
    
    meanAcc(K) = mean(Accuracy);
    Acc{K} = Accuracy;
    
end

meanAcc = max(meanAcc);

function [stPrototypes] = func_SelfTraining(Prototype, LabelVector, K)
%% Do self-training on prototypes
%
%   Prototypes moves towards the K

IDX = knnsearch(LabelVector,Prototype,'Distance','euclidean','K',K);

for p_i = 1:size(Prototype,1)
    stPrototypes(p_i,:) = mean(LabelVector(IDX(p_i,:),:),1);
end

function stPrototypes = func_SelfTraining_Median(Prototype, LabelVector, K)
%% Do median self-training on prototypes
%
%   Prototypes moves towards the K

[IDX,D] = knnsearch(LabelVector,Prototype,'Distance','euclidean','K',K);

for p_i = 1:size(Prototype,1)
    idx = round(K/2);
    stPrototypes(p_i,:) = LabelVector(IDX(p_i,idx),:);
end
