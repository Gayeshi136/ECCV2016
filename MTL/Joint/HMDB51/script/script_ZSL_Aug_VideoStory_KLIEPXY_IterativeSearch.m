%% script to run 50 indepedent datasplits for ZSL
clear;
global D idx_HMDB51 idx_UCF101 idx_OlympicSports idx_CCV tr_LabelVec_HMDB51 tr_LabelVec_UCF101 tr_LabelVec_OlympicSports tr_LabelVec_CCV;

addpath('/import/geb-experiments/Alex/ICCV15/code/SharedTools/function/');
addpath('/import/geb-experiments/Alex/Matlab/Techniques/poblano_toolbox');
addpath('/import/geb-experiments/Alex/ECCV16/code/MTL/SharedTools/VideoStory/function');


%
Para.perc_TrainingSet = 0.5;
Para.perc_TestingSet = 1 - Para.perc_TrainingSet;
Para.cluster_type = 'vlfeat';
Para.nSample = 256000;
Para.CodebookSize = 128;
Para.process = 'org'; % preprocess of dataset: org,sta
Para.FEATURETYPE = 'HOF|HOG|MBH';
Para.nPCA = 0;
SelfTraining = 0;   % Indicator if do selftraining
trial = 1;
Para.EmbeddingMethod = 'add';

Para.lambdaD_range = [1e-3 1e-4 1e-5];
Para.lambdaA_range = [1e-3 1e-4 1e-5];
Para.lambdaS_range = [1e-3 1e-4 1e-5];

Para.lambdaD_range = [1e-3 1e-4 1e-5];
Para.lambdaA_range = [1e-3 1e-4 1e-5];
Para.lambdaS_range = [1e-3 1e-4 1e-5];
Para.LatentDim_range = [50 60 65 70];

% Para.lambdaD_range = [1e-4 1e-5];
% Para.lambdaA_range = [1e-4 1e-5];
% Para.lambdaS_range = [1e-4 1e-5];
% Para.LatentDim_range = 60;

alpha = 0.2;
SelfTraining = 0;

%% Internal Parameters
multishot_base_path = '/import/vision-datasets2/HMDB51/hmdb51_multishot/';
feature_data_base_path = '/import/geb-experiments-archive/Alex/HMDB51/ITF/FV/jointcodebook/';
zeroshot_base_path = '/import/geb-experiments-archive/Alex/HMDB51/ITF/FV/Zeroshot/jointcodebook/Embedding/';

DETECTOR = 'ITF'; % DETECTOR type: STIP, DenseTrj
norm_flag = 1;   % normalization strategy: org,histnorm,zscore

%%% Determine which feature is included
ind = 1;
rest = Para.FEATURETYPE;
while true
    [FeatureTypeList{ind},rest] = strtok(rest,'|');
    if isempty(rest)
        break;
    end
    ind = ind+1;
end


zeroshot_base_path = '/import/geb-experiments-archive/Alex/UCF101/ITF/FV/Zeroshot/jointcodebook/Embedding/';
regression_path = sprintf('%s',zeroshot_base_path);
model_path = sprintf([regression_path,'DatasetSplit_tr-%.1f_ts-%.1f/%s/MTLRegression/RMTL/'],Para.perc_TrainingSet,Para.perc_TestingSet,Para.FEATURETYPE);
if ~exist(model_path,'dir')
    mkdir(model_path);
end

%% Load Label Word Vector Representation
wordvec_path = '/import/vision-datasets2/HMDB51/hmdb51_wordvec/';
temp = load(sprintf([wordvec_path,'ClassLabelPhraseDict_mth-%s.mat'],Para.EmbeddingMethod));
Para.phrasevec_mat = temp.phrasevec_mat;
Para.ClassLabelsPhrase = temp.ClassLabelsPhrase;

%% Load Dataset Info
temp = load([multishot_base_path,'DataSplit.mat']);
Para.sample_class_ind = temp.data_split{1};
Para.ClassNoPerVideo = temp.ClassNoPerVideo;

%% Precompute Distance Matrix
Kernel = 'linear';   % name for kernel we used

if isempty(D)
    
    kernel_path = '/import/geb-experiments-archive/Alex/RegressionTransfer/MergeData/Kernel/';
    kernel_filepath = sprintf([kernel_path,'AugmentedDistMatrix_t-%s_s-%.0g_c-%d_p-%s_n-%d_descr-%s_alpha-%.2f.mat'],...
        Para.cluster_type,Para.nSample,Para.CodebookSize,Para.process,norm_flag,Para.FEATURETYPE,alpha);
    
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


model_path = '/import/geb-experiments-archive/Alex/Joint/HMDB51/VideoStory/Model/';
if ~exist(model_path,'dir')
    mkdir(model_path);
end

perf_path = '/import/geb-experiments-archive/Alex/Joint/HMDB51/VideoStory/Perf/';
if ~exist(perf_path,'dir')
    mkdir(perf_path);
end

%%% Importance Weighting
weight_model_path = '/import/geb-experiments-archive/Alex/ImportanceWeighting/HMDB51/KLIEP_XY/Model/';
weight_model_path = '/import/geb-experiments-archive/Alex/ImportanceWeighting/HMDB51/KLIEP_XY/Model/';

Para.sigmaX = 0.15;
Para.sigmaY = 0.3;

% Para.sigmaX = 0.15;
% Para.sigmaY = 0.3;

bestAcc = 0;
best_lambdaD = 1e-4;
best_lambdaA = 1e-4;
best_lambdaS = 1e-4;

Para.MaxItr = 3;

for trial = 1:5
    
    for itr = 1:MaxItr
        
        %% Search for best lambdaD
        for lambdaD = Para.lambdaD_range
            
            model_filepath = sprintf([model_path,...
                'VideoStory_trial-%d_lambdaD-%g_lambdaA-%g_lambdaS-%g_LatDim-%d_sigX-%g_sigY-%g_aug-1_KLEIPXY_ccv_weight.mat'],...
                trial,lambdaD,best_lambdaA,best_lambdaS,best_LatentDim,Para.sigmaX,Para.sigmaY);
            if ~exist(model_filepath,'file')
                fid = fopen(model_filepath,'w');
                fclose(fid);
                fprintf('Start %s\n',model_filepath);
                script_Train_VideoStory_Weighted;
                script_Test_VideoStory_Weighted;
            else
                fprintf('Exist %s\n',model_filepath);
                script_Test_VideoStory_Weighted;
                continue;
            end
            
            if bestAcc<AccLat
                best_lambdaD = lambdaD;
                bestAcc = AccLat;
            end
            
        end
        
        %% Search for best lambdaD
        for lambdaA = Para.lambdaA_range
            
            model_filepath = sprintf([model_path,...
                'VideoStory_trial-%d_lambdaD-%g_lambdaA-%g_lambdaS-%g_LatDim-%d_sigX-%g_sigY-%g_aug-1_KLEIPXY_ccv_weight.mat'],...
                trial,best_lambdaD,lambdaA,best_lambdaS,best_LatentDim,Para.sigmaX,Para.sigmaY);
            if ~exist(model_filepath,'file')
                fid = fopen(model_filepath,'w');
                fclose(fid);
                fprintf('Start %s\n',model_filepath);
                script_Train_VideoStory_Weighted;
                script_Test_VideoStory_Weighted;
            else
                fprintf('Exist %s\n',model_filepath);
                script_Test_VideoStory_Weighted;
                continue;
            end
            
            if bestAcc<AccLat
                best_lambdaA = lambdaA;
                bestAcc = AccLat;
            end
            
        end
        
        %% Search for best lambdaD
        for lambdaS = Para.lambdaS_range
            
            model_filepath = sprintf([model_path,...
                'VideoStory_trial-%d_lambdaD-%g_lambdaA-%g_lambdaS-%g_LatDim-%d_sigX-%g_sigY-%g_aug-1_KLEIPXY_ccv_weight.mat'],...
                trial,best_lambdaD,best_lambdaS,lambdaS,best_LatentDim,Para.sigmaX,Para.sigmaY);
            if ~exist(model_filepath,'file')
                fid = fopen(model_filepath,'w');
                fclose(fid);
                fprintf('Start %s\n',model_filepath);
                script_Train_VideoStory_Weighted;
                script_Test_VideoStory_Weighted;
            else
                fprintf('Exist %s\n',model_filepath);
                script_Test_VideoStory_Weighted;
                continue;
            end
            
            if bestAcc<AccLat
                best_lambdaS = lambdaS;
                bestAcc = AccLat;
            end
            
        end
        
        %% Search for best latent
        for lambdaS = Para.LatentDim_range
            
            model_filepath = sprintf([model_path,...
                'VideoStory_trial-%d_lambdaD-%g_lambdaA-%g_lambdaS-%g_LatDim-%d_sigX-%g_sigY-%g_aug-1_KLEIPXY_ccv_weight.mat'],...
                trial,best_lambdaD,best_lambdaS,best_lambdaS,LatentDim,Para.sigmaX,Para.sigmaY);
            if ~exist(model_filepath,'file')
                fid = fopen(model_filepath,'w');
                fclose(fid);
                fprintf('Start %s\n',model_filepath);
                script_Train_VideoStory_Weighted;
                script_Test_VideoStory_Weighted;
            else
                fprintf('Exist %s\n',model_filepath);
                script_Test_VideoStory_Weighted;
                continue;
            end
            
            if bestAcc<AccLat
                best_LatentDim = LatentDim;
                bestAcc = AccLat;
            end
            
        end
        
    end
    
end