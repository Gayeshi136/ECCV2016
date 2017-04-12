%% script to run 50 indepedent datasplits for ZSL
clear;
global D idx_HMDB51 idx_UCF101 idx_OlympicSports idx_CCV tr_LabelVec_HMDB51 tr_LabelVec_UCF101 tr_LabelVec_OlympicSports tr_LabelVec_CCV;

addpath('../function');
addpath('/import/geb-experiments/Alex/Matlab/Techniques/poblano_toolbox');
addpath('/import/geb-experiments/Alex/ECCV16/code/MTL/SharedTools/GOMTL/function');

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
lambdaS_range = [1e-4 1e-5 1e-6 1e-7 1e-8];
lambdaA_range = [1e-2 1e-3 1e-4];
LatentProportion_range = [0.15 0.2 0.25];
alpha = 0.2;
Para.SelfTraining = 0;
Para.DataAugmentation = 1;

%% Internal Parameters
feature_data_base_path = '/import/geb-experiments-archive/Alex/UCF101/ITF/FV/jointcodebook/';
datasplit_path = '/import/vision-datasets3/vd3_Alex/Dataset/UCF101/THUMOS13/ucfTrainTestlist/';
zeroshot_datasplit_path = [datasplit_path,'zeroshot/'];
labelvector_path = [datasplit_path,'LabelVector/'];

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

load(sprintf([labelvector_path,'ClassNameVector_mth-%s.mat'],EmbeddingMethod));
temp = load([datasplit_path,'VideoNamesPerClass.mat']);
Para.ClassNoPerVideo = temp.ClassNoPerVideo;
Para.phrasevec_mat = phrasevec_mat;
clear phrasevec_mat temp;

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

model_path = sprintf('/import/geb-experiments-archive/Alex/MTL/UCF101/GOMTL/Model_Aug-%d/',Para.DataAugmentation);
if ~exist(model_path,'dir')
    mkdir(model_path);
end

perf_path = sprintf('/import/geb-experiments-archive/Alex/MTL/UCF101/GOMTL/Perf_Aug-%d/',Para.DataAugmentation);
if ~exist(perf_path,'dir')
    mkdir(perf_path);
end

Perf = {};

for lambdaS = lambdaS_range
    for lambdaA = lambdaA_range
        for prop = LatentProportion_range
            
            Para.lambdaS = lambdaS;
            Para.lambdaA = lambdaA;
            Para.LatentProportion = prop;
            
            
            
            %%% Check if model is computed
            model_filepath = sprintf([model_path,'LinearRegress_trial-%d_embed-%s_lambdaS-%g_lambdaA-%g_LatProp-%g_task-dim_aug-%d.mat'],...
                trial,EmbeddingMethod,Para.lambdaS,Para.lambdaA,Para.LatentProportion,Para.DataAugmentation);
            if ~exist(model_filepath,'file')
                fprintf('Doesn''t exist %s\n',model_filepath);
                continue;
            else
                try temp=load(model_filepath);
                catch
                    fprintf('Corrupted or Unfinished %s\n',model_filepath);
                end
                Model = temp.Model;
                fprintf('Exist %s\n',model_filepath);
                
            end
            
            %% Load Data Split
            load(sprintf([zeroshot_datasplit_path,'DatasetSplit_tr-%.1f_ts-%.1f_t-%d.mat'],perc_TrainingSet,perc_TestingSet,trial),'idx_TrainingSet','idx_TestingSet');
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
            
            
            %% Generate Testing Kernel Matrix
            selected_tr_idx = Para.tr_sample_ind'.*idx_UCF101;
            selected_tr_idx = selected_tr_idx(selected_tr_idx~=0);
            aug_tr_idx = [idx_HMDB51 selected_tr_idx idx_OlympicSports idx_CCV];            selected_tr_idx = selected_tr_idx(selected_tr_idx~=0);
            Para.selected_tr_idx = aug_tr_idx;
            selected_ts_idx = Para.ts_sample_ind'.*idx_UCF101;
            Para.selected_ts_idx = selected_ts_idx(selected_ts_idx~=0);
            
            
            %% Test GOMTL Model
            [meanAcc,Acc] = func_ts_ZSL_GOMTL_Dim_Embed(Para,Model);
            
            Perf = [Perf;{Para.lambdaS Para.lambdaA prop meanAcc}];
            
        end
    end
end
