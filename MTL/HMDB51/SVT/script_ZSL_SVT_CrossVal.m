%% script to run 50 indepedent datasplits for ZSL
clear;
global D idx_HMDB51 idx_UCF101 idx_OlympicSports idx_CCV tr_LabelVec_HMDB51 tr_LabelVec_UCF101 tr_LabelVec_OlympicSports tr_LabelVec_CCV;

addpath('../function');
addpath('/import/geb-experiments/Alex/Matlab/Techniques/poblano_toolbox');
addpath('/import/geb-experiments/Alex/Matlab/Techniques/SVT_MostRecent');

if matlabpool('size')~=5 && matlabpool('size')~=0
    myCluster = parcluster('local');
    delete(myCluster.Jobs);
    matlabpool close;
    matlabpool 5;
    
end


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
% lambdaS_range = [1e-4 1e-5 1e-6 1e-7 1e-8];
lambdaS_range = [1e-4 1e-5 1e-6];
lambdaA_range = [1e-4 1e-5 1e-6];
lambdaD_range = [1e-4 1e-5 1e-6];
LatentDim_range = [50 100 200 300];
alpha = 0.2;
Para.SelfTraining = 0;

%% Internal Parameters
multishot_base_path = '/import/vision-datasets2/HMDB51/hmdb51_multishot/';
feature_data_base_path = '/import/geb-experiments-archive/Alex/HMDB51/ITF/FV/jointcodebook/';
zeroshot_base_path = '/import/geb-experiments-archive/Alex/HMDB51/ITF/FV/Zeroshot/jointcodebook/Embedding/';

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


zeroshot_base_path = '/import/geb-experiments-archive/Alex/UCF101/ITF/FV/Zeroshot/jointcodebook/Embedding/';
regression_path = sprintf('%s',zeroshot_base_path);
model_path = sprintf([regression_path,'DatasetSplit_tr-%.1f_ts-%.1f/%s/MTLRegression/RMTL/'],perc_TrainingSet,perc_TestingSet,FEATURETYPE);
if ~exist(model_path,'dir')
    mkdir(model_path);
end

%% Load Label Word Vector Representation
wordvec_path = '/import/vision-datasets2/HMDB51/hmdb51_wordvec/';
temp = load(sprintf([wordvec_path,'ClassLabelPhraseDict_mth-%s.mat'],EmbeddingMethod));
Para.phrasevec_mat = temp.phrasevec_mat;
ClassLabelsPhrase = temp.ClassLabelsPhrase;

%% Load Dataset Info
temp = load([multishot_base_path,'DataSplit.mat']);
sample_class_ind = temp.data_split{1};
Para.ClassNoPerVideo = temp.ClassNoPerVideo;

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


%% Run 50 random splits
Para.lambdaS = 1e-7;
Para.lambdaA = 1e-3;
Para.LatentProportion = 0.2;    % proportion of latent tasks

model_path = '/import/geb-experiments-archive/Alex/MTL/HMDB51/VideoStory/Model/';
if ~exist(model_path,'dir')
    mkdir(model_path);
end

perf_path = '/import/geb-experiments-archive/Alex/MTL/HMDB51/VideoStory/Perf/';
if ~exist(perf_path,'dir')
    mkdir(perf_path);
end


% for lambdaD = lambdaD_range
%     for lambdaA = lambdaA_range
%         for lambdaS = lambdaS_range
%             for LatentDim = LatentDim_range
%
%                 Para.lambdaD = lambdaD;
%                 Para.lambdaS = lambdaS;
%                 Para.lambdaA = lambdaA;
%                 Para.LatentDim = LatentDim;
%

%%% Check if model is computed
% model_filepath = sprintf([model_path,'VideoStory_trial-%d_lambdaD-%g_lambdaA-%g_lambdaS-%g_LatDim-%d_aug-0.mat'],trial,Para.lambdaD,Para.lambdaA,Para.lambdaS,Para.LatentDim);
% if ~exist(model_filepath,'file')
%     %                     fid = fopen(model_filepath,'w');
%     %                     fclose(fid);
% else
%     fprintf('Exist %s\n',model_filepath);
%     %                     continue;
% end

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

%% Generate Training Kernel Matrix
selected_tr_idx = Para.tr_sample_ind'.*idx_HMDB51;
selected_tr_idx = selected_tr_idx(selected_tr_idx~=0);
aug_tr_idx = [selected_tr_idx];

Z = [tr_LabelVec_HMDB51(Para.tr_sample_ind,:)];

Z = func_L2Normalization(Z)';
Z_sumt = sum(Z,1);
T = size(Z,1); % number of tasks
N = size(Z,2); % number of instances

K = D(aug_tr_idx,aug_tr_idx);

N_L = round(size(Z,1)*Para.LatentProportion);

%% Generate Testing Kernel Matrix
selected_tr_idx = Para.tr_sample_ind'.*idx_HMDB51;
selected_tr_idx = selected_tr_idx(selected_tr_idx~=0);
Para.selected_tr_idx = selected_tr_idx;
selected_ts_idx = Para.ts_sample_ind'.*idx_HMDB51;
Para.selected_ts_idx = selected_ts_idx(selected_ts_idx~=0);

K_ts = D(Para.selected_ts_idx,aug_tr_idx);   %D(selected_ts_idx,aug_tr_idx) * A;

%% Train Low Rank Factorization
Y = [[K Z'];[K_ts zeros(size(K_ts,1),size(Z,1))]];
Omega = [ones(size([K Z'])) ; ones(size(K_ts)) zeros(size(K_ts,1),size(Z,1))];
Omega = reshape(Omega,1,numel(Omega));
Omega = find(Omega == 1);
data = reshape(Y(Omega),1,numel(Omega));

n1 = size(Y,1);
n2 = size(Y,2);

tau = 1*sqrt(n1*n2);
m=numel(data);
p  = m/(n1*n2);
delta = 1.2/p;   
maxiter = 500;
tol = 1e-4;
%% Approximate minimum nuclear norm solution by SVT algorithm
% Note: SVT, as called below, is setup for noiseless data
%   (i.e. equality constraints).

fprintf('\nSolving by SVT...\n');
tic
[U,S,V,numiter] = SVT([n1 n2],Omega,data,tau,delta,maxiter,tol);
toc

X = U*S*V';


%                 [Model.D, Model.A , Model.S , Model.L]=func_KernelizedVideoStory(K,Z,Para.LatentDim,Para.lambdaD,Para.lambdaS,Para.lambdaA);

%     Model = func_tr_ZSL_GOMTL_Dim_ncg(Para);

%%% Save Result
%                 save(model_filepath,'Model');

%             end
%         end
%     end
% end