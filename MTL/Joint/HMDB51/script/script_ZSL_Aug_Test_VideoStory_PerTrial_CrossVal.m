%% script to run 50 indepedent datasplits for ZSL
clear;
global D idx_HMDB51 idx_UCF101 idx_OlympicSports idx_CCV tr_LabelVec_HMDB51 tr_LabelVec_UCF101 tr_LabelVec_OlympicSports tr_LabelVec_CCV;

addpath('../function');
addpath('/import/geb-experiments/Alex/Matlab/Techniques/poblano_toolbox');
addpath('/import/geb-experiments/Alex/ECCV16/code/MTL/SharedTools/VideoStory/function');


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
Para.DataAug = 1;

Para.lambdaD_range = [1e-3 1e-4 1e-5];
Para.lambdaA_range = [1e-3 1e-4 1e-5];
Para.lambdaS_range = [1e-3 1e-4 1e-5];
Para.LatentDim_range = [100 110 120];


Para.lambdaD_range = [1e-3 1e-4 1e-5];
Para.lambdaA_range = [1e-3 1e-4 1e-5];
Para.lambdaS_range = [1e-3 1e-4 1e-5];
Para.LatentDim_range = [50 60 70 80 65 75];

% Para.lambdaD_range = [1e-4 1e-5];
% Para.lambdaA_range = [1e-4 1e-5];
% Para.lambdaS_range = [1e-4 1e-5];
% Para.LatentDim_range = 60;
% Para.lambdaD_range = [1e-4];
% Para.lambdaA_range = [1e-5];
% Para.lambdaS_range = [1e-4];

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
% Para.lambdaS = 1e-7;
% Para.lambdaA = 1e-3;
% Para.LatentProportion = 0.2;    % proportion of latent tasks

model_path = '/import/geb-experiments-archive/Alex/Joint/HMDB51/VideoStory/Model/';
if ~exist(model_path,'dir')
    mkdir(model_path);
end

perf_path = '/import/geb-experiments-archive/Alex/Joint/HMDB51/VideoStory/Perf/';
if ~exist(perf_path,'dir')
    mkdir(perf_path);
end


Para.sigmaX = 0.15;
Para.sigmaY = 0.3;

for trial = 1:5
    
    Perf_Dist = [];
    Perf_Lat = [];
    for lambdaD = Para.lambdaD_range
        for lambdaA = Para.lambdaA_range
            for lambdaS = Para.lambdaS_range
                for LatentDim = Para.LatentDim_range
                    
                    Para.lambdaD = lambdaD;
                    Para.lambdaS = lambdaS;
                    Para.lambdaA = lambdaA;
                    Para.LatentDim = LatentDim;
                    
                    %% Check if performance is evaluated
                    perf_filepath = fullfile(perf_path,sprintf('Perf_aug-%d_trial-%d_D-%g_A-%g_S-%g_lat-%d_X-%g_Y-%g_weight.mat',...
                        Para.DataAug,trial,Para.lambdaD,Para.lambdaS,Para.lambdaA,Para.LatentDim,Para.sigmaX,Para.sigmaY));
                    
                    if ~exist(perf_filepath,'file')
                        fid = fopen(perf_filepath,'w');
                        fclose(fid);
                    else
                        corrupted = 0;
                        try load(perf_filepath);
                        catch
                            
                            corrupted = 1;
                        end
                        if ~corrupted
%                             Perf_Dist = [Perf_Dist;[Para.lambdaD Para.lambdaA Para.lambdaS Para.LatentDim AccDist]];
%                             fprintf('trial=%d lambdaD=%g lambdaA=%g lambdaS=%g Lat=%d\n distributed acc=%.2f\n',...
%                                 trial,Para.lambdaD,Para.lambdaA,Para.lambdaS,Para.LatentDim,100*AccDist);
                            
                            Perf_Lat = [Perf_Lat;[Para.lambdaD Para.lambdaA Para.lambdaS Para.LatentDim AccLat]];
                            fprintf('trial=%d lambdaD=%g lambdaA=%g lambdaS=%g Lat=%d\n latent acc=%.2f\n\n',...
                                trial,Para.lambdaD,Para.lambdaA,Para.lambdaS,Para.LatentDim,100*AccLat);
                            
                            continue;
                        end
                        
                    end
                    
                    
                    %% Check if model is computed
                    model_filepath = sprintf([model_path,...
                        'VideoStory_trial-%d_lambdaD-%g_lambdaA-%g_lambdaS-%g_LatDim-%d_sigX-%g_sigY-%g_aug-1_KLEIPXY_ccv_weight.mat'],...
                        trial,lambdaD,lambdaA,lambdaS,LatentDim,Para.sigmaX,Para.sigmaY);
                    if ~exist(model_filepath,'file')
                        fprintf('Doesn''t exist %s\n',model_filepath);
                        continue;
                    else
                        try temp=load(model_filepath);
                        catch
                                                         fprintf('Corrupted or Unfinished %s\n',model_filepath);
                            continue;
                        end
                        Model = temp.Model;
                                                 fprintf('Exist %s\n',model_filepath);
                        
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
                    Para.selected_tr_idx = [ selected_tr_idx idx_UCF101 idx_OlympicSports idx_CCV];
                    selected_ts_idx = Para.ts_sample_ind'.*idx_HMDB51;
                    Para.selected_ts_idx = selected_ts_idx(selected_ts_idx~=0);
                    
                    %% Test VideoStory Model
                    
                    %                     [AccDist,Acc] = func_ts_ZSL_VideoStory_ACC_zscore_Distributed(Para,Model);
                    %
                    %                     Perf_Dist = [Perf_Dist;[Para.lambdaD Para.lambdaA Para.lambdaS Para.LatentDim AccDist]];
                    %                     fprintf('trial=%d lambdaD=%g lambdaA=%g lambdaS=%g Lat=%d\n distributed acc=%.2f\n',...
                    %                         trial,Para.lambdaD,Para.lambdaA,Para.lambdaS,Para.LatentDim,100*AccDist);
                    
                    [AccLat,Acc] = func_ts_ZSL_VideoStory_ACC_zscore_Latent(Para,Model);
                    
                    Perf_Lat = [Perf_Lat;[Para.lambdaD Para.lambdaA Para.lambdaS Para.LatentDim AccLat]];
                    fprintf('trial=%d lambdaD=%g lambdaA=%g lambdaS=%g Lat=%d\n latent acc=%.2f\n\n',...
                        trial,Para.lambdaD,Para.lambdaA,Para.lambdaS,Para.LatentDim,100*AccLat);
                    
                    
                    %% Save Performance
                    save(perf_filepath,'AccLat');
                    
                    
                end
            end
        end
        
    end
    
    %             [PerfDistributed.bestAccTrial,indx] = max(Perf_Dist(:,5));
    %             PerfDistributed.bestLambdaD = Perf_Dist(indx,1);
    %             PerfDistributed.bestLambdaA = Perf_Dist(indx,2);
    %             PerfDistributed.bestLambdaS = Perf_Dist(indx,3);
    %             PerfDistributed.bestLatentDim = Perf_Dist(indx,4);
    
    [PerfLatent.bestAccTrial,indx] = max(Perf_Lat(:,5));
    PerfLatent.bestLambdaD = Perf_Lat(indx,1);
    PerfLatent.bestLambdaA = Perf_Lat(indx,2);
    PerfLatent.bestLambdaS = Perf_Lat(indx,3);
    PerfLatent.bestLatentDim = Perf_Lat(indx,4);
    
    %             bestAccTrial_Dist(trial) = PerfDistributed.bestAccTrial;
    bestAccTrial_Lat(trial) = PerfLatent.bestAccTrial;
    bestAccTrial_D(trial) = PerfLatent.bestLambdaD;
    bestAccTrial_A(trial) = PerfLatent.bestLambdaA;
    bestAccTrial_S(trial) = PerfLatent.bestLambdaS;
    bestAccTrial_L(trial) = PerfLatent.bestLatentDim;

    %% save results
    
    %     save(perf_filepath,'PerfDistributed','PerfLatent');
end


%     mean(bestAccTrial_Dist)
%     std(bestAccTrial_Dist)

mean(bestAccTrial_Lat)
std(bestAccTrial_Lat)