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
lambda0_range = [1e-4 1e-3 1e-2 1e-1 1e0 1e1];
lambdat_range = [1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1e0 1e1];
alpha = 0.2;
Para.SelfTraining = 0;

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


%% Do 50 trials
Para.lambda0 = 0.01;
Para.lambdat = 0.1;

model_path = '/import/geb-experiments-archive/Alex/MTL/UCF101/RMTL/Model/';
if ~exist(model_path,'dir')
    mkdir(model_path);
end

perf_path = '/import/geb-experiments-archive/Alex/MTL/UCF101/RMTL/Perf/';
if ~exist(perf_path,'dir')
    mkdir(perf_path);
end


for trial = 1:50
    
    %% Load Data Split
    datasplit_path = '/import/vision-datasets3/vd3_Alex/Dataset/UCF101/THUMOS13/ucfTrainTestlist/';
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
    
    %%% Check if model is computed
    model_filepath = sprintf([model_path,'LinearRegress_trial-%d_embed-%s_lambda0-%g_lambdat-%g_task-dim_Aug-0.mat'],trial,EmbeddingMethod,Para.lambda0,Para.lambdat);
    if ~exist(model_filepath,'file')
        fid = fopen(model_filepath,'w');
        fclose(fid);
        fprintf('Start %s\n',model_filepath);
        
        
        selected_tr_idx = Para.tr_sample_ind'.*idx_UCF101;
        selected_tr_idx = selected_tr_idx(selected_tr_idx~=0);
        Para.aug_tr_idx = [selected_tr_idx];
        
        Para.Z = [tr_LabelVec_UCF101(Para.tr_sample_ind,:)];
        %% Train RMTL Model
        Model = func_tr_ZSL_RMTL_Dim(Para);
        
        %%% Save Regression Model
        save(model_filepath,'Model');
        
    else
        
        try Model = load(model_filepath);
        catch
            fprintf('Corrupted or Unfinished %s\n',model_filepath);
            continue;
        end
        
        fprintf('Exist %s\n',model_filepath);
        
        Model = Model.Model;
        
        perf_filepath = sprintf([perf_path,'meanAcc_trial-%d_lambda0-%g_lambdat-%g_latent-1.mat'],trial,Para.lambda0,Para.lambdat);
        if ~exist(perf_filepath,'file')
            
            %% Test RMTL Model
            selected_tr_idx = Para.tr_sample_ind'.*idx_UCF101;            
            selected_tr_idx = selected_tr_idx(selected_tr_idx~=0);
            Para.aug_tr_idx = [selected_tr_idx];
            
            selected_ts_idx = Para.ts_sample_ind'.*idx_UCF101;
            Para.selected_ts_idx = selected_ts_idx(selected_ts_idx~=0);
            
            clear selected_ts_idx selected_tr_idx;
                      
            [meanAcc(trial),Acc{trial}] = func_ts_ZSL_RMTL_Dim_Latent(Para,Model);
            
            %%% Save Result
            save(perf_filepath,'meanAcc','Acc');
            fprintf('%dth trial acc = %.2f\n',trial,meanAcc(trial)*100);
        else
            continue;
        end
    end
    
end

