%% function to train VS on HMDB51

function func_Tr_VS_HMDB(Para)

global D idx_HMDB51 idx_UCF101 idx_OlympicSports idx_CCV tr_LabelVec_HMDB51 tr_LabelVec_UCF101 tr_LabelVec_OlympicSports tr_LabelVec_CCV;

for lambdaD = Para.lambdaD_range
    for lambdaA = Para.lambdaA_range
        for lambdaS = Para.lambdaS_range
            for LatentDim = Para.LatentDim_range
                
                Para.lambdaD = lambdaD;
                Para.lambdaS = lambdaS;
                Para.lambdaA = lambdaA;
                Para.LatentDim = LatentDim;
                
                
                %%% Check if model is computed
                model_filepath = sprintf([model_path,'VideoStory_trial-%d_lambdaD-%g_lambdaA-%g_lambdaS-%g_LatDim-%d_aug-0.mat'],trial,Para.lambdaD,Para.lambdaA,Para.lambdaS,Para.LatentDim);
                if ~exist(model_filepath,'file')
                    fid = fopen(model_filepath,'w');
                    fclose(fid);
                else
                    fprintf('Exist %s\n',model_filepath);
                    %                     continue;
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
                
                %% Generate Training Kernel Matrix
                selected_tr_idx = Para.tr_sample_ind'.*idx_HMDB51;
                selected_tr_idx = selected_tr_idx(selected_tr_idx~=0);
                aug_tr_idx = [selected_tr_idx];
                
                Z = [tr_LabelVec_HMDB51(Para.tr_sample_ind,:)];
                
                Z = func_L2Normalization(Z)';
                Z_sumt = sum(Z,1);
                T = size(Z,1); % number of tasks
                N = size(Z,2); % number of instances
                
                K = D(aug_tr_idx,aug_tr_idx);
                
                N_L = round(size(Z,1)*Para.LatentProportion);
                %% Train GOMTL Model
                
                [Model.D, Model.A , Model.S , Model.L]=func_KernelizedVideoStory(K,Z,Para.LatentDim,Para.lambdaD,Para.lambdaS,Para.lambdaA);
                
                %     Model = func_tr_ZSL_GOMTL_Dim_ncg(Para);
                
                %%% Save Result
                save(model_filepath,'Model');
                
            end
        end
    end
end