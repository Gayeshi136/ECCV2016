%% script to run 50 indepedent datasplits for ZSL
clear;
global D idx_HMDB51 idx_UCF101 idx_OlympicSports idx_CCV tr_LabelVec_HMDB51 tr_LabelVec_UCF101 tr_LabelVec_OlympicSports tr_LabelVec_CCV;

addpath('../function');
addpath('/import/geb-experiments/Alex/Matlab/Techniques/poblano_toolbox');
addpath('/import/geb-experiments/Alex/ECCV16/code/MTL/SharedTools/GOMTL_WI/function');
addpath('/import/geb-experiments/Alex/ICCV15/code/SharedTools/function');

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
lambdaS_range = [1e-6 1e-7 1e-8];
lambdaA_range = [1e-3];
LatentProportion_range = [0.4 0.5];
alpha = 0.2;
Para.SelfTraining = 0;
Para.DataAugmentation = 1;

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
% model_path = sprintf([regression_path,'DatasetSplit_tr-%.1f_ts-%.1f/%s/MTLRegression/RMTL/'],perc_TrainingSet,perc_TestingSet,FEATURETYPE);
% if ~exist(model_path,'dir')
%     mkdir(model_path);
% end

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
Para.lambdaS = 1e-6;
Para.lambdaA = 1e-4;
Para.LatentProportion = 0.2;    % proportion of latent tasks
Para.SpreadCoeff = 0.05;

model_path = sprintf('/import/geb-experiments-archive/Alex/MTL/HMDB51/GOMTL_WI/Model_Aug-%d/',Para.DataAugmentation);
if ~exist(model_path,'dir')
    mkdir(model_path);
end

perf_path = sprintf('/import/geb-experiments-archive/Alex/MTL/HMDB51/GOMTL_WI/Perf_Aug-%d/',Para.DataAugmentation);
if ~exist(perf_path,'dir')
    mkdir(perf_path);
end

for trial = 1:50
    
    %%% Check if model is computed
    model_filepath = sprintf([model_path,'LinearRegress_trial-%d_embed-%s_lambdaS-%g_lambdaA-%g_LatProp-%g_task-dim_aug-%d_WI-%g.mat'],...
        trial,EmbeddingMethod,Para.lambdaS,Para.lambdaA,Para.LatentProportion,Para.DataAugmentation,Para.SpreadCoeff);
    if ~exist(model_filepath,'file')
        fid = fopen(model_filepath,'w');
        fclose(fid);
    else
        fprintf('Exist %s\n',model_filepath);
        continue;
    end
    
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
    Para.aug_tr_idx = [ selected_tr_idx idx_UCF101 idx_OlympicSports idx_CCV];
    Para.aux_tr_idx = [idx_UCF101 idx_OlympicSports idx_CCV];
    
    selected_ts_idx = Para.ts_sample_ind'.*idx_HMDB51;
    Para.selected_ts_idx = selected_ts_idx(selected_ts_idx~=0);
    
    Z = [tr_LabelVec_HMDB51(Para.tr_sample_ind,:) ; tr_LabelVec_UCF101 ; tr_LabelVec_OlympicSports ; tr_LabelVec_CCV];
    
    Z = func_L2Normalization(Z)';
    Z_sumt = sum(Z,1);
    T = size(Z,1); % number of tasks
    N = size(Z,2); % number of instances
    
    K = D(Para.aug_tr_idx,Para.aug_tr_idx);
    
    N_L = round(size(Z,1)*Para.LatentProportion);
    
    
    %% Compute Auxiliary instance weight
    D_tr_ts = 1-D(selected_tr_idx,Para.selected_ts_idx)/3;
    mean_min_D_tr_ts = mean(min(D_tr_ts,[],2));
    
    D_aux_ts = 1-D(Para.aux_tr_idx,Para.selected_ts_idx)/3;
    % mean(min(D_aux_ts,[],2))
    
    minD_aug_ts = min([D_tr_ts;D_aux_ts],[],2);
    
    S_tr = func_L2Normalization(tr_LabelVec_HMDB51(Para.tr_sample_ind,:));
    S_ts = func_L2Normalization(tr_LabelVec_HMDB51(Para.ts_sample_ind,:));
    S_aux = func_L2Normalization([tr_LabelVec_UCF101 ; tr_LabelVec_OlympicSports ; tr_LabelVec_CCV]);
    Ds_tr_ts = 1-S_tr*S_ts';
    mean_min_Ds_tr_ts = mean(min(Ds_tr_ts,[],2));
    Ds_aux_ts = 1-S_aux*S_ts';
    minDs_aug_ts = min([Ds_tr_ts ; Ds_aux_ts],[],2);
    
    AugInstanceWeight = (mean_min_D_tr_ts+mean_min_Ds_tr_ts)./(minD_aug_ts+minDs_aug_ts);
    AugInstanceWeight_center = AugInstanceWeight-mean(AugInstanceWeight);
    Para.AugInstanceWeight = sign(AugInstanceWeight_center).*abs(AugInstanceWeight_center).^Para.SpreadCoeff + mean(AugInstanceWeight);
    clear AugInstanceWeight;
    
    %% Train GOMTL Model
    
    [Model.S , Model.A , Model.L]=func_KernelizedGOMTLWI_admm_ncg(K,Z,N_L,Para);
    
    %%% Save Result
    save(model_filepath,'Model');
    
end

