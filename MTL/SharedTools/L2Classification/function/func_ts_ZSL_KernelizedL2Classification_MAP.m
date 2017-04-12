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

function [map,ap] = func_ts_ZSL_KernelizedL2Classification_MAP(K,V,Y,Para,Model)

% global D;

addpath('/import/geb-experiments/Alex/CVPR15/Code/Zeroshot/HMDBregression/Code/Basic/FVNormalization');
addpath('/import/geb-experiments/Alex/ICCV15/code/TransferRegression/CollectData/function');
addpath('/import/geb-experiments/Alex/ICCV15/code/SharedTools/function/');

%% Testing Ridge Regression model
Y_hat = V'*Model.A*K;
[~,predict_ClassNo] = max(Y_hat,[],1);

%% Calculate Mean Average Precision
confs = Y_hat;

for c_i = 1:length(Para.idx_TestingSet)
    labelsVid = Y == Para.idx_TestingSet(c_i);
    ap(c_i) = func_CalcMap(confs(c_i,:), labelsVid);
    
end

map = mean(ap);
