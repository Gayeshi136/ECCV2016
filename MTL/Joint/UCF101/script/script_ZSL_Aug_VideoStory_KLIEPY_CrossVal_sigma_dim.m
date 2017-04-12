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

Para.lambdaD_range = [1e-3];
Para.lambdaA_range = [1e-5];
Para.lambdaS_range = [1e-3];
Para.LatentDim_range = [75 80 85];
Para.sigma_range = [0.1 0.2 0.25 0.3 0.35 0.4];

% Para.lambdaD_range = [1e-4 1e-5];
% Para.lambdaA_range = [1e-4 1e-5];
% Para.lambdaS_range = [1e-4 1e-5];
% Para.LatentDim_range = 60;

alpha = 0.2;
SelfTraining = 0;

%% Internal Parameters
feature_data_base_path = '/import/geb-experiments-archive/Alex/UCF101/ITF/FV/jointcodebook/';
datasplit_path = '/import/vision-datasets3/vd3_Alex/Dataset/UCF101/THUMOS13/ucfTrainTestlist/';
zeroshot_datasplit_path = [datasplit_path,'zeroshot/'];
labelvector_path = [datasplit_path,'LabelVector/'];

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

load(sprintf([labelvector_path,'ClassNameVector_mth-%s.mat'],Para.EmbeddingMethod));
temp = load([datasplit_path,'VideoNamesPerClass.mat']);
Para.ClassNoPerVideo = temp.ClassNoPerVideo;
Para.phrasevec_mat = phrasevec_mat;
clear phrasevec_mat temp;

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


model_path = '/import/geb-experiments-archive/Alex/Joint/UCF101/VideoStory/KLIEP_Y/Model/';
if ~exist(model_path,'dir')
    mkdir(model_path);
end

perf_path = '/import/geb-experiments-archive/Alex/Joint/UCF101/VideoStory/KLIEP_Y/Perf/';
if ~exist(perf_path,'dir')
    mkdir(perf_path);
end

%%% Importance Weighting
weight_model_path = '/import/geb-experiments-archive/Alex/ImportanceWeighting/UCF101/KLIEP_Y/Weight/';

% Para.sigmaX = 0.15;
% Para.sigmaY = 0.3;

% Para.sigmaX = 0.15;
% Para.sigmaY = 0.3;

for trial = 1
    for lambdaD = Para.lambdaD_range
        for lambdaA = Para.lambdaA_range
            for lambdaS = Para.lambdaS_range
                for sigmaY = Para.sigma_range
                    for LatentDim = Para.LatentDim_range
                        Para.sigmaY = sigmaY;
                        %%% Check if model is computed
                        model_filepath = sprintf([model_path,...
                            'VideoStory_trial-%d_lambdaD-%g_lambdaA-%g_lambdaS-%g_LatDim-%d_sigY-%g_aug-1_KLEIPY_ccv_weight.mat'],...
                            trial,lambdaD,lambdaA,lambdaS,LatentDim,Para.sigmaY);
                        if ~exist(model_filepath,'file')
                            fid = fopen(model_filepath,'w');
                            fclose(fid);
                            fprintf('Start %s\n',model_filepath);
                            
                        else
                            fprintf('Exist %s\n',model_filepath);
                            continue;
                        end
                        
                        
                        %% Load Data Split
                        datasplit_path = '/import/vision-datasets3/vd3_Alex/Dataset/UCF101/THUMOS13/ucfTrainTestlist/';
                        load(sprintf([zeroshot_datasplit_path,'DatasetSplit_tr-%.1f_ts-%.1f_t-%d.mat'],Para.perc_TrainingSet,Para.perc_TestingSet,trial),'idx_TrainingSet','idx_TestingSet');
                        Para.idx_TrainingSet = sort(idx_TrainingSet,'ascend');
                        Para.idx_TestingSet = sort(idx_TestingSet,'ascend');
                        clear idx_TrainingSet idx_TestingSet;
                        
                        %% Prepare Training Data
                        tr_LabelVec = [];
                        Para.tr_sample_ind = zeros(size(Para.ClassNoPerVideo,1),1);   % train sample index
                        %%% Normalize label vector
                        SS = sum(Para.phrasevec_mat.^2,2);
                        label_k = sqrt(size(Para.phrasevec_mat,2)./SS);
                        Para.phrasevec_mat = repmat(label_k,1,size(Para.phrasevec_mat,2)) .* Para.phrasevec_mat;
                        
                        for c_tr = 1:length(Para.idx_TrainingSet)
                            
                            %% Extract Training Features for each class
                            class_no = Para.idx_TrainingSet(c_tr);
                            tr_sample_class_ind = Para.ClassNoPerVideo == class_no;
                            tr_LabelVec = [tr_LabelVec ; repmat(Para.phrasevec_mat(class_no,:),sum(tr_sample_class_ind),1)];
                            Para.tr_sample_ind = Para.tr_sample_ind + tr_sample_class_ind;
                        end
                        
                        Para.tr_sample_ind = logical(Para.tr_sample_ind);
                        Para.ts_sample_ind = ~Para.tr_sample_ind;
                        
                        %% Load weight
                        weight_filepath = sprintf([weight_model_path,'Weight_trial-%d_embed-%s_sigma-%g_aug-1_norm-zscore_ccv.mat'],trial,Para.EmbeddingMethod,Para.sigmaY);
                        KLIEP_Y = load(weight_filepath,'weight');
                        weight = KLIEP_Y.weight;
                        %% Load Weight
                        %                     XYmodel_filepath = sprintf([weight_model_path,'LinearRegress_trial-%d_embed-%s_sigmaX-%g_sigmaY-%g_aug-1.mat'],trial,Para.EmbeddingMethod,Para.sigmaX,Para.sigmaY);
                        %                     KLIEP_XY = load(XYmodel_filepath,'Model');
                        %                     weight = KLIEP_XY.Model.weight;
                        
                        %% Generate Training Kernel Matrix
                        selected_tr_idx = Para.tr_sample_ind'.*idx_UCF101;
                        selected_tr_idx = selected_tr_idx(selected_tr_idx~=0);
                        aug_tr_idx = [idx_HMDB51 selected_tr_idx  idx_OlympicSports idx_CCV];
                        
                        Z_All = zscore(func_L2Normalization([tr_LabelVec_HMDB51 ; tr_LabelVec_UCF101 ; tr_LabelVec_OlympicSports ; tr_LabelVec_CCV]));
                        Z = Z_All(aug_tr_idx,:)';
                        
                        %                     K = D(aug_tr_idx,aug_tr_idx)*sparse(1:N_l,1:N_l,sqrt(weight));
                        K = D(aug_tr_idx,aug_tr_idx);
                        
                        %% Train GOMTL Model
                        param.MaxItr = 30;
                        param.epsilon = 5e-3;
                        [Model.D, Model.A , Model.S , Model.L]=func_KernelizedVideoStory(K,Z,LatentDim,lambdaD,lambdaS,lambdaA,param);
                        
                        %%% Save Result
                        save(model_filepath,'Model');
                    end
                end
            end
        end
    end
end