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

function [meanAcc,Accuracy] = func_ts_ZSL_Attr_zscore(varargin)

global D idx_HMDB51 idx_UCF101 idx_OlympicSports idx_CCV tr_LabelVec_HMDB51 tr_LabelVec_UCF101 tr_LabelVec_OlympicSports tr_LabelVec_CCV;

addpath('/import/geb-experiments/Alex/CVPR15/Code/Zeroshot/HMDBregression/Code/Basic/FVNormalization');

%% Parse Input
if isempty(varargin)
    perc_TrainingSet = 0.5;
    perc_TestingSet = 1 - perc_TrainingSet;
    cluster_type = 'vlfeat';
    nSample = 256000;
    CodebookSize = 128;
    process = 'org'; % preprocess of dataset: org,sta
    FEATURETYPE = 'HOF|HOG|MBH';
    nPCA = 0;
    SelfTraining = 0;   % Indicator if do selftraining
    trial = 1;
    EmbeddingMethod = 'add'; % add multiply combine
else
    if nargin >=1
        perc_TrainingSet = varargin{1};
    else
        perc_TrainingSet = 0.5; % Training set percentage
    end
    perc_TestingSet = 1 - perc_TrainingSet;
    
    if nargin >=2
        cluster_type = varargin{2};
    else
        cluster_type = 'vlfeat';
    end
    
    if nargin >=3
        nSample = varargin{3};
    else
        nSample = 1e5;
    end
    
    if nargin >=4
        CodebookSize = varargin{4};
    else
        CodebookSize = 128;
    end
    
    if nargin >=5
        process = varargin{5};
    else
        process = 'org';
    end
    
    if nargin >=6
        FEATURETYPE = varargin{6};
    else
        
        FEATURETYPE = 'HOF|HOG|MBH';
    end
    
    if nargin >=7
        nPCA = varargin{7};
    else
        nPCA = 0;
        
    end
    
    if nargin >=8
        C = varargin{8};
    else
        C = 0;
        
    end
    
    if nargin >=9
        SelfTraining = varargin{9};
    else
        SelfTraining = 0;
        
    end
    
    if nargin >=10
        trial = varargin{10};
    else
        trial = 0;
        
    end
    
    if nargin >=11
        EmbeddingMethod = varargin{11};
    else
        EmbeddingMethod = 'add';
        
    end
    
    if nargin >=12
        lambda = varargin{12};
    else
        lambda = 1e-6;
    end
end

%% Function Internal Parameters
feature_data_base_path = '/import/geb-experiments-archive/Alex/UCF101/ITF/FV/jointcodebook/';
zeroshot_base_path = '/import/geb-experiments-archive/Alex/UCF101/ITF/FV/Zeroshot/jointcodebook/Embedding/';
if ~exist(zeroshot_base_path,'dir')
    mkdir(zeroshot_base_path);
end

datasplit_path = '/import/vision-datasets3/vd3_Alex/Dataset/UCF101/THUMOS13/ucfTrainTestlist/';
zeroshot_datasplit_path = [datasplit_path,'zeroshot/'];
labelvector_path = [datasplit_path,'LabelVector/'];
data_base_path = '/import/geb-experiments-archive/Alex/UCF101/';
attr_base_path = [data_base_path 'Attribute/'];

DETECTOR = 'ITF'; % DETECTOR type: STIP, DenseTrj
norm_flag = 1;   % normalization strategy: org,histnorm,zscore

%%% Determine which feature is included
ind = 1;
rest = FEATURETYPE;
while true
    [FeatureTypeList{ind},rest] = strtok(rest,'|');
    if isempty(rest)
        break;
    end
    ind = ind+1;
end

%% Parameter to Tune
if ~exist('alpha','var')
    alpha = 0.2;
end
if ~exist('lambda','var')
    lambda = 1e-6;
end

fprintf('percentage of training set: %.2f\n technique: %s\n number of kmeans centres: %d\n preprocess of dataset: %s\n',...
    perc_TrainingSet,cluster_type,CodebookSize,process);

%% Load Data Split

datasplit_path = '/import/vision-datasets3/vd3_Alex/Dataset/UCF101/THUMOS13/ucfTrainTestlist/';

load(sprintf([zeroshot_datasplit_path,'DatasetSplit_tr-%.1f_ts-%.1f_t-%d.mat'],perc_TrainingSet,perc_TestingSet,trial));

idx_TrainingSet = sort(idx_TrainingSet,'ascend');
idx_TestingSet = sort(idx_TestingSet,'ascend');

%% Load Label Word Vector Representation
% 
% load(sprintf([labelvector_path,'ClassNameVector_mth-%s.mat'],EmbeddingMethod));
% load([datasplit_path,'VideoNamesPerClass.mat']);.

%% Load Dataset Info
load(sprintf([labelvector_path,'ClassNameVector_mth-%s.mat'],EmbeddingMethod));
temp = load([datasplit_path,'VideoNamesPerClass.mat']);
ClassNoPerVideo = temp.ClassNoPerVideo;

%% Load Attributes
load([attr_base_path,'fixed_class_attributes_UCF101.mat'],'class_attributes');
%%% Construct training and testing attributes
phrasevec_mat = double([class_attributes.annotations]');


%% Prepare Training Data
tr_LabelVec = [];
tr_sample_ind = zeros(size(ClassNoPerVideo,1),1);   % train sample index

%%% Normalize label vector
SS = sum(phrasevec_mat.^2,2);
label_k = sqrt(size(phrasevec_mat,2)./SS);
phrasevec_mat = repmat(label_k,1,size(phrasevec_mat,2)) .* phrasevec_mat;
idx_TrainingSet = sort(idx_TrainingSet,'ascend');

for c_tr = 1:length(idx_TrainingSet)
    
    %% Extract Training Features for each class
    class_no = idx_TrainingSet(c_tr);
    tr_sample_class_ind = ClassNoPerVideo == class_no;
    tr_LabelVec = [tr_LabelVec ; repmat(phrasevec_mat(class_no,:),sum(tr_sample_class_ind),1)];
    tr_sample_ind = tr_sample_ind + tr_sample_class_ind;
end

tr_sample_ind = logical(tr_sample_ind);
ts_sample_ind = ~tr_sample_ind;

ts_ClassNo = ClassNoPerVideo(ts_sample_ind);

%% Precompute Distance Matrix
Kernel = 'linear';   % name for kernel we used

if isempty(D)
    
    kernel_path = '/import/geb-experiments-archive/Alex/RegressionTransfer/MergeData/Kernel/';
    kernel_filepath = sprintf([kernel_path,'AugmentedDistMatrix_t-%s_s-%.0g_c-%d_p-%s_n-%d_descr-%s_alpha-%.2f.mat'],...
        cluster_type,nSample,CodebookSize,process,norm_flag,FEATURETYPE,alpha);
    
    if exist(kernel_filepath,'file')
        
        %%% Load precompute Kernel
        load(kernel_filepath);
        
    else
        DataType = 'all';
        
        %% Load Auxiliary Dataset
        [FVFeature_HMDB51,tr_LabelVec_HMDB51]=func_CollectHMDB51(DataType);
        [FVFeature_UCF101,tr_LabelVec_UCF101]=func_CollectUCF101(DataType);
        [FVFeature_OlympicSports,tr_LabelVec_OlympicSports]=func_CollectOlympicSports(DataType);
        [FVFeature_CCV,tr_LabelVec_CCV]=func_CollectCCV(DataType);
        
        idx_HMDB51 = 1:size(FVFeature_HMDB51,1);
        idx_UCF101 = idx_HMDB51(end)+1:idx_HMDB51(end)+size(FVFeature_UCF101,1);
        idx_OlympicSports = idx_UCF101(end)+1:idx_UCF101(end)+size(FVFeature_OlympicSports,1);
        idx_CCV = idx_OlympicSports(end)+1:idx_OlympicSports(end)+size(FVFeature_CCV,1);
        
        all_FeatureMat = [FVFeature_HMDB51 ; FVFeature_UCF101 ; FVFeature_OlympicSports ; FVFeature_CCV];
        
        D = func_PrecomputeKernel(all_FeatureMat,all_FeatureMat,'linear');
        
        save(kernel_filepath,'D','idx_HMDB51','idx_UCF101','idx_OlympicSports','idx_CCV',...
            'tr_LabelVec_HMDB51','tr_LabelVec_UCF101','tr_LabelVec_OlympicSports','tr_LabelVec_CCV','-v7.3');
        
    end
end

ts_sample_no = ts_sample_ind'.*idx_UCF101;
ts_sample_no = ts_sample_no(ts_sample_no~=0);

tr_sample_no = tr_sample_ind'.*idx_UCF101;
tr_sample_no = tr_sample_no(tr_sample_no~=0);

%%% Generate Linear Kernel Matrix
K_ts = D(ts_sample_no,tr_sample_no);

%% Testing Ridge Regression model

regression_path = sprintf('%s',zeroshot_base_path);
model_path = sprintf([regression_path,'DatasetSplit_tr-%.1f_ts-%.1f/%s/'],perc_TrainingSet,perc_TestingSet,FEATURETYPE);

load(sprintf([model_path,'LinearRegress_trial-%d_embed-%s_lambda-%g_Attr_norm-zscore.mat'],trial,EmbeddingMethod,lambda),'A');

S_ts = K_ts * A;
S_ts = zscore(S_ts);

%% Knn to predict final labels

%%% Generate Prototypes
Prototype = zscore((phrasevec_mat));
Prototype = Prototype(idx_TestingSet,:);

if SelfTraining
    %% Self-training
    parfor K = 1:200;
        stPrototype = func_SelfTraining(Prototype, S_ts, K);
        
        stPrototype = func_SelfTraining(stPrototype, S_ts, K);
        
        
        %%% Predict labels
        predict_ClassNo = knnsearch(stPrototype,S_ts,'Distance','cosine');
        
        %%% Calculate Average Precision for each Class
        Accuracy = zeros(1,length(idx_TestingSet));
        for c_ts = 1:length(idx_TestingSet)
            
            currentClass = idx_TestingSet(c_ts);
            currentClass_SampleIndex = ts_ClassNo==currentClass;
            currentClass_Predict = predict_ClassNo(currentClass_SampleIndex);   % predicted class no
            Accuracy(c_ts) = sum(currentClass_Predict == c_ts)/length(currentClass_Predict);
            
        end
        
        meanAcc(K) = mean(Accuracy);
    end
else
    %%% Predict labels
    predict_ClassNo = knnsearch(Prototype,S_ts,'Distance','euclidean');
    
    %%% Calculate Average Precision for each Class
    for c_ts = 1:length(idx_TestingSet)
        
        currentClass = idx_TestingSet(c_ts);
        currentClass_SampleIndex = ts_ClassNo==currentClass;
        currentClass_Predict = predict_ClassNo(currentClass_SampleIndex);   % predicted class no
        Accuracy(1,c_ts) = sum(currentClass_Predict == c_ts)/length(currentClass_Predict);
        
    end
    
    meanAcc = mean(Accuracy);
end

% meanAcc = max(meanAcc);

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
