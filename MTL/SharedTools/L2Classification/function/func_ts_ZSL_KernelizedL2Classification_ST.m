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

function [meanAcc,Accuracy] = func_ts_ZSL_KernelizedL2Classification_ST(K,V,Y,Para,Model)

% global D;

addpath('/import/geb-experiments/Alex/CVPR15/Code/Zeroshot/HMDBregression/Code/Basic/FVNormalization');
addpath('/import/geb-experiments/Alex/ICCV15/code/TransferRegression/CollectData/function');
addpath('/import/geb-experiments/Alex/ICCV15/code/SharedTools/function/');

%% Self-Training
ProjectData = Model.A*K;
ProjectData = func_L2Normalization(ProjectData')';
Knn = 100;
V_ST = func_SelfTraining(V', ProjectData', Knn);

%% Testing ESZSL model after ST
Y_hat = V_ST*Model.A*K;
% Y_hat = V_ST*ProjectData;

 [~,predict_ClassNo] = max(Y_hat,[],1);

%% Calculate Average Accuracy for each Class
for c_ts = 1:length(Para.idx_TestingSet)
    
    currentClass = Para.idx_TestingSet(c_ts);
    currentClass_SampleIndex = Y==currentClass;
    currentClass_Predict = predict_ClassNo(currentClass_SampleIndex);   % predicted class no
    Accuracy(1,c_ts) = sum(currentClass_Predict == c_ts)/length(currentClass_Predict);
    
end

meanAcc = mean(Accuracy);


% Y_hat = V'*Model.A*K

% Y_hat = V'*Model.A*K;
% 
% [~,predict_ClassNo] = max(Y_hat,[],1);
% %%% Predict labels
% % predict_ClassNo = knnsearch(Prototype,S_ts,'Distance','cosine');
% 
% %%% Calculate Average Precision for each Class
% for c_ts = 1:length(Para.idx_TestingSet)
%     
%     currentClass = Para.idx_TestingSet(c_ts);
%     currentClass_SampleIndex = Y==currentClass;
%     currentClass_Predict = predict_ClassNo(currentClass_SampleIndex);   % predicted class no
%     Accuracy(1,c_ts) = sum(currentClass_Predict == c_ts)/length(currentClass_Predict);
%     
% end
% 
% meanAcc = mean(Accuracy);

function [stPrototypes] = func_SelfTraining(Prototype, LabelVector, K)
%% Do self-training on prototypes
%
%   Prototypes moves towards the K

% IDX = knnsearch(LabelVector,Prototype,'Distance','euclidean','K',K);

score = Prototype*LabelVector';

for c = 1:size(Prototype,1)
    [~,sort_idx] = sort(score(c,:),'descend');
    IDX(c,:) = sort_idx(1:K);
end

for p_i = 1:size(Prototype,1)
    stPrototypes(p_i,:) = mean(LabelVector(IDX(p_i,:),:),1);
end