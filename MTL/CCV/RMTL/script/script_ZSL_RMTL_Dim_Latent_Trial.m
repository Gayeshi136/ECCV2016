%% script to run 50 indepedent datasplits for ZSL
clear;

global D_CCV idx_HMDB51 idx_UCF101 idx_OlympicSports idx_CCV  idx_CCV_All tr_LabelVec_HMDB51 tr_LabelVec_UCF101 tr_LabelVec_OlympicSports tr_LabelVec_CCV;

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
feature_data_base_path = '/import/geb-experiments-archive/Alex/CCV/ITF/FV/';
zeroshot_base_path = '/import/geb-experiments-archive/Alex/CCV/ITF/FV/Model/Zeroshot/jointcodebook/Embedding/';
if ~exist(zeroshot_base_path,'dir')
    mkdir(zeroshot_base_path);
end

datasplit_path = '/import/geb-experiments-archive/Alex/CCV/DataSplit/';
zeroshot_datasplit_path = [datasplit_path,'Zeroshot/'];
labelvector_path = '/import/geb-experiments-archive/Alex/CCV/Embedding/Word2Vec/';


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
temp = load('/import/geb-experiments-archive/Alex/CCV/DataSplit/Multishot/DataSplit.mat');
Para.ClassNoPerVideo = temp.DataSplit.ClassLabelPerVideo;

%% Precompute Distance Matrix
Kernel = 'linear';   % name for kernel we used

if isempty(D_CCV)
    
    kernel_path = '/import/geb-experiments-archive/Alex/RegressionTransfer/MergeData/Kernel/';
    kernel_filepath = sprintf([kernel_path,'CCVAugmentedDistMatrix_t-%s_s-%.0g_c-%d_p-%s_n-%d_descr-%s_alpha-%.2f.mat'],...
        cluster_type,nSample,CodebookSize,process,norm_flag,FEATURETYPE,alpha);
    
    if exist(kernel_filepath,'file')
        
        %%% Load precompute Kernel
        load(kernel_filepath);
        D_CCV = D;
    else
        DataType = 'all';
        
        %% Load Auxiliary Dataset
        [FVFeature_HMDB51,tr_LabelVec_HMDB51]=func_CollectHMDB51(DataType);
        [FVFeature_UCF101,tr_LabelVec_UCF101]=func_CollectUCF101(DataType);
        [FVFeature_OlympicSports,tr_LabelVec_OlympicSports]=func_CollectOlympicSports(DataType);
        [FVFeature_CCV_All,tr_LabelVec_CCV_All]=func_CollectCCV_All(DataType);
        
        idx_HMDB51 = 1:size(FVFeature_HMDB51,1);
        idx_UCF101 = idx_HMDB51(end)+1:idx_HMDB51(end)+size(FVFeature_UCF101,1);
        idx_OlympicSports = idx_UCF101(end)+1:idx_UCF101(end)+size(FVFeature_OlympicSports,1);
        idx_CCV_All = idx_OlympicSports(end)+1:idx_OlympicSports(end)+size(FVFeature_CCV_All,1);
        
        
        all_FeatureMat = [FVFeature_HMDB51 ; FVFeature_UCF101 ; FVFeature_OlympicSports ; FVFeature_CCV_All];
        
        D_CCV = func_PrecomputeKernel(all_FeatureMat,all_FeatureMat,'linear');
        
        save(kernel_filepath,'D_CCV','idx_HMDB51','idx_UCF101','idx_OlympicSports','idx_CCV_All',...
            'tr_LabelVec_HMDB51','tr_LabelVec_UCF101','tr_LabelVec_OlympicSports','tr_LabelVec_CCV_All','-v7.3');
        
    end
    
end

%% Do 50 trials
Para.lambda0 = 0.01;
Para.lambdat = 0.1;

model_path = '/import/geb-experiments-archive/Alex/MTL/CCV/RMTL/Model/';
if ~exist(model_path,'dir')
    mkdir(model_path);
end

perf_path = '/import/geb-experiments-archive/Alex/MTL/CCV/RMTL/Perf/';
if ~exist(perf_path,'dir')
    mkdir(perf_path);
end

for trial = 1:50
    
    %%% Zeroshot Datasplit
    load(sprintf([zeroshot_datasplit_path,'DatasetSplit_tr-%.1f_ts-%.1f_t-%d.mat'],perc_TrainingSet,perc_TestingSet,trial));
    
    Para.idx_TrainingSet = sort(idx_TrainingSet,'ascend');
    Para.idx_TestingSet = sort(idx_TestingSet,'ascend');
    clear idx_TrainingSet idx_TestingSet;
    
    %% Prepare Training Data
    tr_LabelVec = [];
    tr_sample_ind = [];
    all_sample_ind = [];
    for v_i = 1:size(Para.ClassNoPerVideo,1)  
        positive_classno = find(Para.ClassNoPerVideo(v_i,:)==1);
        %%% collect all data with positive labels
        if sum(positive_classno)
            all_sample_ind = [all_sample_ind v_i * ones(1,length(positive_classno))];
        end
        if sum(ismember(positive_classno,Para.idx_TestingSet))
            continue;
        end
        if sum(ismember(positive_classno,Para.idx_TrainingSet))
            tr_sample_ind = [tr_sample_ind v_i * ones(1,length(positive_classno))];
            tr_LabelVec = [tr_LabelVec ; Para.phrasevec_mat(positive_classno,:)];
            
        end
    end
    Para.tr_sample_ind = logical(tr_sample_ind);
    Para.ts_sample_ind = ~tr_sample_ind;
    clear tr_sample_ind ts_sample_ind;
    
    
    %%% Check if model is computed
    model_filepath = sprintf([model_path,'LinearRegress_trial-%d_embed-%s_lambda0-%g_lambdat-%g_task-dim_Aug-0.mat'],trial,EmbeddingMethod,Para.lambda0,Para.lambdat);
    if ~exist(model_filepath,'file')
        fid = fopen(model_filepath,'w');
        fclose(fid);
        fprintf('Start %s\n',model_filepath);   
        selected_tr_idx = idx_CCV_All(Para.tr_sample_ind);
        selected_tr_idx = selected_tr_idx(selected_tr_idx~=0);
        Para.aug_tr_idx = [selected_tr_idx];
        
        Para.Z = [tr_LabelVec];
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
        
        perf_filepath = sprintf([perf_path,'map_trial-%d_lambda0-%g_lambdat-%g_latent-1.mat'],trial,Para.lambda0,Para.lambdat);
        if ~exist(perf_filepath,'file')

            
            %% Prepare Testing Data
            % ts_sample_ind = zeros(size(DataSplit.ClassLabelPerVideo,1),1);   % train sample index
            ts_sample_ind = [];
            tr_sample_ind = [];
            all_sample_ind = [];
            for v_i = 1:size(Para.ClassNoPerVideo,1)
                
                positive_classno = find(Para.ClassNoPerVideo(v_i,:)==1);
                %%% collect all data with positive labels
                if sum(positive_classno)
                    all_sample_ind = [all_sample_ind v_i * ones(1,length(positive_classno))];
                end
                
                if isempty(positive_classno)
                    ts_sample_ind = [ts_sample_ind v_i];
                    continue;
                end
                
                if sum(ismember(positive_classno,Para.idx_TestingSet))
                    ts_sample_ind = [ts_sample_ind v_i];
                    continue;
                end
                
                if sum(ismember(positive_classno,Para.idx_TrainingSet))
                    tr_sample_ind = [tr_sample_ind v_i * ones(1,length(positive_classno))];
                end
                
            end
            Para.tr_sample_ind = logical(tr_sample_ind);
            Para.ts_sample_ind = ts_sample_ind;
            clear tr_sample_ind ts_sample_ind;
            
            %% Test RMTL Model
            selected_tr_idx = idx_CCV_All(Para.tr_sample_ind);
            selected_tr_idx = selected_tr_idx(selected_tr_idx~=0);
            Para.aug_tr_idx = [selected_tr_idx];
            
            selected_ts_idx = idx_CCV_All(Para.ts_sample_ind);
            Para.selected_ts_idx = selected_ts_idx(selected_ts_idx~=0);
            clear selected_ts_idx selected_tr_idx;
            
            [map(trial),ap{trial}] = func_ts_ZSL_RMTL_MAP_Dim_Latent_CCV(Para,Model);
            
            %%% Save Result
            save(perf_filepath,'map','ap');
            fprintf('%dth trial map = %.2f\n',trial,map(trial)*100);
        else
            continue;
        end
    end
    
end

