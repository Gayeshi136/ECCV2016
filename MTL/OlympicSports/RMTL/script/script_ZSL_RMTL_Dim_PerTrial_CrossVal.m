%% script to run 50 indepedent datasplits for ZSL
clear;

global D idx_HMDB51 idx_UCF101 idx_OlympicSports idx_CCV tr_LabelVec_HMDB51 tr_LabelVec_UCF101 tr_LabelVec_OlympicSports tr_LabelVec_CCV;

addpath('../function');
addpath('/import/geb-experiments/Alex/ECCV16/code/MTL/SharedTools/function');

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
lambda0_range = [1e-3 1e-2 1e-1 1];
lambdat_range = [1e-3 1e-2 1e-1 1];
alpha = 0.2;
Para.SelfTraining = 0;

%% Internal Parameters
feature_data_base_path = '/import/geb-experiments-archive/Alex/OlympicSports/FV_ITF/';
zeroshot_base_path = '/import/geb-experiments-archive/Alex/OlympicSports/FV_ITF/Model/Zeroshot/jointcodebook/Embedding/';
if ~exist(zeroshot_base_path,'dir')
    mkdir(zeroshot_base_path);
end

datasplit_path = '/import/geb-experiments-archive/Alex/OlympicSports/DataSplit/';
zeroshot_datasplit_path = [datasplit_path,'Zeroshot/'];
labelvector_path = '/import/geb-experiments-archive/Alex/OlympicSports/Embedding/Word2Vec/';


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
temp = load(sprintf([labelvector_path,'ClassLabelPhraseDict_mth-%s.mat'],EmbeddingMethod));
Para.phrasevec_mat = temp.phrasevec_mat;

%% Load Dataset Info
temp = load('/import/geb-experiments-archive/Alex/OlympicSports/DataSplit/Multishot/DataSplit.mat');
Para.ClassNoPerVideo = temp.DataSplit.ClassNoPerVideo;

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


model_path = '/import/geb-experiments-archive/Alex/MTL/OlympicSports/RMTL/Model/';
if ~exist(model_path,'dir')
    mkdir(model_path);
end

perf_path = '/import/geb-experiments-archive/Alex/MTL/OlympicSports/RMTL/Perf/';
if ~exist(perf_path,'dir')
    mkdir(perf_path);
end

%% Grid Search para
for trial = 1:5
    for lambda0 = lambda0_range
        for lambdat = lambdat_range
            
            Para.lambda0 = lambda0;
            Para.lambdat = lambdat;
            
            %%% Check if model is computed
            model_filepath = sprintf([model_path,'RMTL_trial-%d_lambda0-%g_lambdat-%g_aug-0_norm-zscore.mat'],trial,Para.lambda0,Para.lambdat);
            if ~exist(model_filepath,'file')
                fid = fopen(model_filepath,'w');
                fclose(fid);
                fprintf('Start %s\n',model_filepath);
                
            else
                fprintf('Exist %s\n',model_filepath);
                continue;
            end
            
            %%% Zeroshot Datasplit
            load(sprintf([zeroshot_datasplit_path,'DatasetSplit_tr-%.1f_ts-%.1f_t-%d.mat'],perc_TrainingSet,perc_TestingSet,trial));
            
            Para.idx_TrainingSet = sort(idx_TrainingSet,'ascend');
            Para.idx_TestingSet = sort(idx_TestingSet,'ascend');
            clear idx_TrainingSet idx_TestingSet;
            
            %% Prepare Training Data
            tr_sample_ind = zeros(size(Para.ClassNoPerVideo,1),1);   % train sample index
            for c_tr = 1:length(Para.idx_TrainingSet)
                
                %% Extract Training Features for each class
                class_no = Para.idx_TrainingSet(c_tr);
                tr_sample_class_ind = Para.ClassNoPerVideo == class_no;
                tr_sample_ind = tr_sample_ind + tr_sample_class_ind;
            end
            
            Para.tr_sample_ind = logical(tr_sample_ind);
            Para.ts_sample_ind = ~tr_sample_ind;
            clear tr_sample_ind ts_sample_ind;
            
            selected_tr_idx = Para.tr_sample_ind'.*idx_OlympicSports;
            selected_tr_idx = selected_tr_idx(selected_tr_idx~=0);
            Para.aug_tr_idx = [selected_tr_idx];
            
            Z_all = zscore(func_L2Normalization(tr_LabelVec_OlympicSports));
            Para.Z = Z_all(Para.tr_sample_ind,:);
            %% Train RMTL Model
            Model = func_tr_ZSL_RMTL_Dim_zscore(Para);
            
            %%% Save Regression Model
            save(model_filepath,'Model');
            
        end
    end
end
