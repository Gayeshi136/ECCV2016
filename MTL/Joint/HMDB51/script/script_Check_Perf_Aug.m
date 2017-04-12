%% script to check VS perf on Augmented Training Data


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

Para.lambdaD_range = [1e-4 1e-5];
Para.lambdaA_range = [ 1e-4 1e-5];
Para.lambdaS_range = [ 1e-4 1e-5];
Para.LatentDim_range = [50 80 110 120 130];

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

model_path = '/import/geb-experiments-archive/Alex/Joint/HMDB51/VideoStory/Model/';
if ~exist(model_path,'dir')
    mkdir(model_path);
end

perf_path = '/import/geb-experiments-archive/Alex/Joint/HMDB51/VideoStory/Perf/';
if ~exist(perf_path,'dir')
    mkdir(perf_path);
end


bestAccTrial_Dist = [];
bestAccTrial_Lat = [];

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
                    perf_filepath = fullfile(perf_path,sprintf('Perf_aug-%d_trial-%d_D-%g_A-%g_S-%g_lat-%d.mat',...
                        Para.DataAug,trial,Para.lambdaD,Para.lambdaS,Para.lambdaA,Para.LatentDim));
                    
                    if ~exist(perf_filepath,'file')
                        continue;
                    else
                        corrupted = 0;
                        try load(perf_filepath);
                        catch
                            
                            corrupted = 1;
                        end
                        if corrupted
                            continue;
                        end
                        
                    end
                    
                    
                    Perf_Dist = [Perf_Dist;[Para.lambdaD Para.lambdaA Para.lambdaS Para.LatentDim AccDist]];
                    Perf_Lat = [Perf_Lat;[Para.lambdaD Para.lambdaA Para.lambdaS Para.LatentDim AccLat]];
                    
                    
                    
                end
            end
        end
        
    end
    
    [PerfDistributed.bestAccTrial,indx] = max(Perf_Dist(:,5));
    PerfDistributed.bestLambdaD = Perf_Dist(indx,1);
    PerfDistributed.bestLambdaA = Perf_Dist(indx,2);
    PerfDistributed.bestLambdaS = Perf_Dist(indx,3);
    PerfDistributed.bestLatentDim = Perf_Dist(indx,4);
    
    [PerfLatent.bestAccTrial,indx] = max(Perf_Lat(:,5));
    PerfLatent.bestLambdaD = Perf_Lat(indx,1);
    PerfLatent.bestLambdaA = Perf_Lat(indx,2);
    PerfLatent.bestLambdaS = Perf_Lat(indx,3);
    PerfLatent.bestLatentDim = Perf_Lat(indx,4);
    
    bestAccTrial_Dist(trial) = PerfDistributed.bestAccTrial;
    bestLatTrial_Dist(trial) = PerfDistributed.bestLatentDim;
    
    bestAccTrial_Lat(trial) = PerfLatent.bestAccTrial;
    bestLatTrial_Lat(trial) = PerfLatent.bestLatentDim;
    
end

mean(bestAccTrial_Dist)
std(bestAccTrial_Dist)

mean(bestAccTrial_Lat)
std(bestAccTrial_Lat)