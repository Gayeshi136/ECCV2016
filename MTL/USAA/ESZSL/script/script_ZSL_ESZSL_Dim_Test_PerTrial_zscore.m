%% script to run 50 indepedent datasplits for ZSL
clear;

global D_CCV idx_HMDB51 idx_UCF101 idx_OlympicSports idx_CCV_All idx_USAA tr_LabelVec_HMDB51 tr_LabelVec_UCF101 tr_LabelVec_OlympicSports tr_LabelVec_CCV;

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
lambda0_range = [1e-4 1e-3 1e-2 1e-1 1];
lambdat_range = [1e-4 1e-3 1e-2 1e-1 1];
alpha = 0.2;
Para.SelfTraining = 0;

%% Function Internal Parameters
feature_data_base_path = '/import/geb-experiments-archive/Alex/USAA/ITF/FV/';
zeroshot_base_path = '/import/geb-experiments-archive/Alex/USAA/ITF/FV/Model/Zeroshot/jointcodebook/Embedding/';
if ~exist(zeroshot_base_path,'dir')
    mkdir(zeroshot_base_path);
end

datasplit_path = '/import/geb-experiments-archive/Alex/USAA/DataSplit/';
zeroshot_datasplit_path = [datasplit_path,'Zeroshot/'];
labelvector_path = '/import/geb-experiments-archive/Alex/USAA/Embedding/Word2Vec/';

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


%% Load Label Word Vector Representation
% wordvec_path = '/import/vision-datasets2/HMDB51/hmdb51_wordvec/';
% temp = load(sprintf([wordvec_path,'ClassLabelPhraseDict_mth-%s.mat'],EmbeddingMethod));
% Para.phrasevec_mat = temp.phrasevec_mat;
% ClassLabelsPhrase = temp.ClassLabelsPhrase;
%
% %% Load Dataset Info
% temp = load([multishot_base_path,'DataSplit.mat']);
% sample_class_ind = temp.data_split{1};
% Para.ClassNoPerVideo = temp.ClassNoPerVideo;


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


%% Load Dataset Info
load('/import/geb-experiments-archive/Alex/USAA/input.mat','test_video_name','train_video_name');
test_video_name = test_video_name + 4659;
idx_USAA = [train_video_name ; test_video_name] + idx_CCV_All(1)-1;

Para.lambda0 = 1e-3;
Para.lambdat = 1;

model_path = '/import/geb-experiments-archive/Alex/MTL/USAA/RMTL/Model/';
if ~exist(model_path,'dir')
    mkdir(model_path);
end

perf_path = '/import/geb-experiments-archive/Alex/MTL/USAA/RMTL/Perf/';
if ~exist(perf_path,'dir')
    mkdir(perf_path);
end

Perf = {};

%% Grid Search para
% for lambda0 = lambda0_range
%     for lambdat = lambdat_range
%
%         Para.lambda0 = lambda0;
%         Para.lambdat = lambdat;
% for trial = 1:3
%     Perf_Dist = [];
%     Perf_Lat = [];
    for lambda0 = lambda0_range
        for lambdat = lambdat_range
            
            Para.lambda0 = lambda0;
            Para.lambdat = lambdat;
            %%% Check if model is computed
            model_filepath = sprintf([model_path,'RMTL_trial-%d_lambda0-%g_lambdat-%g_aug-0_norm-zscore.mat'],trial,Para.lambda0,Para.lambdat);
            if ~exist(model_filepath,'file')
                fprintf('Doesn'' Exist %s\n',model_filepath);
                continue;
            else
                fprintf('Exist %s\n',model_filepath);
                try load(model_filepath,'Model');
                catch
                    fprintf('Corrupted %s\n',model_filepath);
                    continue;
                end
            end
            
            %% Load Data Split
            
            %%% Zeroshot Datasplit
            temp = load(sprintf([zeroshot_datasplit_path,'DatasetSplit_t-%d.mat'],trial));
            Para.ClassNoPerVideo = temp.ClassNoPerVideo;
            idx_TrainingSet = sort(temp.idx_TrainingSet,'ascend');
            idx_TestingSet = sort(temp.idx_TestingSet,'ascend');
            
            %%% General Datasplit
            load('/import/geb-experiments-archive/Alex/USAA/input.mat','train_video_label','test_video_label');
            
            % load('/import/geb-experiments-archive/Alex/CCV/DataSplit/Multishot/DataSplit.mat');
            
            %% Load Label Word Vector Representation
            
            load(sprintf([labelvector_path,'ClassLabelPhraseDict_mth-%s.mat'],EmbeddingMethod));
            % load([datasplit_path,'VideoNamesPerClass.mat']);
            Para.phrasevec_mat = func_L2Normalization(phrasevec_mat);
            
            %% Prepare Training Data
            tr_LabelVec = [];
            ts_LabelVec = [];
            
            tr_sample_ind = zeros(length(train_video_label)+length(test_video_label),1);   % train sample index
            tr_sample_ind(1:length(train_video_label)) = ismember(train_video_label,idx_TrainingSet);
            tr_sample_ind = logical(tr_sample_ind);
            tr_LabelVec = [];
            for v_i = 1:size(train_video_label,1)
                if sum(ismember(train_video_label(v_i),idx_TrainingSet))
                    tr_LabelVec = [tr_LabelVec ; phrasevec_mat(train_video_label(v_i),:)];
                end
                
            end
            
            for v_i = 1:size(test_video_label,1)
                if sum(ismember(test_video_label(v_i),idx_TestingSet))
                    ts_LabelVec = [ts_LabelVec ; phrasevec_mat(test_video_label(v_i),:)];
                end
                
            end
            
            ts_sample_ind = zeros(length(train_video_label)+length(test_video_label),1);   % test sample index
            ts_sample_ind(length(train_video_label)+1:length(train_video_label)+length(test_video_label)) =...
                ismember(test_video_label,idx_TestingSet);
            ts_sample_ind = logical(ts_sample_ind);
            
            selected_tr_idx = tr_sample_ind'.*idx_USAA';
            selected_tr_idx = selected_tr_idx(selected_tr_idx~=0);
            selected_ts_idx = ts_sample_ind'.*idx_USAA';
            selected_ts_idx = selected_ts_idx(selected_ts_idx~=0);
            
            %% Test RMTL
            Para.aug_tr_idx =selected_tr_idx;
            Para.selected_ts_idx =selected_ts_idx;
            Para.ts_sample_ind =ts_sample_ind;
            Para.idx_TestingSet = idx_TestingSet;
            
            [meanAcc,Accuracy] = func_ts_ZSL_RMTL_ACC_Dim_USAA_Distributed(Para,Model);
            
            Perf_Dist = [Perf_Dist;[Para.lambda0 Para.lambdat meanAcc]];
            fprintf('lambda0=%g lambdat=%g map=%.2f\n',...
                Para.lambda0,Para.lambdat,100*meanAcc);
            
            [meanAcc,Accuracy] = func_ts_ZSL_RMTL_ACC_Dim_USAA_Latent(Para,Model);
            
            Perf_Lat = [Perf_Lat;[Para.lambda0 Para.lambdat meanAcc]];
            fprintf('lambda0=%g lambdat=%g map=%.2f\n',...
                Para.lambda0,Para.lambdat,100*meanAcc);
            
            %         Perf = [Perf ; {lambda0 lambdat map}];
            %         fprintf('lambda0=%g lambdat=%g map=%.2f\n',...
            %             lambda0,lambdat,100*map);
            %     end
        end
    end
    
%     [PerfDistributed.bestAccTrial(trial),indx] = max(Perf_Dist(:,3));
%     PerfDistributed.bestLambda0(trial) = Perf_Dist(indx,1);
%     PerfDistributed.bestLambdat(trial) = Perf_Dist(indx,2);
%     
%     [PerfLatent.bestAccTrial(trial),indx] = max(Perf_Lat(:,3));
%     PerfLatent.bestLambda0(trial) = Perf_Lat(indx,1);
%     PerfLatent.bestLambdat(trial) = Perf_Lat(indx,2);
%     
% end
% 
% mean(PerfDistributed.bestAccTrial)
% std(PerfDistributed.bestAccTrial)
% 
% mean(PerfLatent.bestAccTrial)
% std(PerfLatent.bestAccTrial)
% 
% %% save results
% perf_filepath = fullfile(perf_path,'bestPerf.mat');
% 
% save(perf_filepath,'PerfDistributed','PerfLatent');