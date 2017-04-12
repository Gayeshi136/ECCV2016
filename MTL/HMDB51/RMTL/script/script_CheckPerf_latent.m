%% sript to check performance
Dataset = 'HMDB51'; % HMDB51, UCF101, OlympicSports, CCV

perf_path = sprintf('/import/geb-experiments-archive/Alex/MTL/%s/RMTL/Perf/',Dataset);

TrialAcc = zeros(1,50);

for trial = 1:50
   
    perf_filepath = sprintf([perf_path,'meanAcc_trial-%d_lambda0-%g_lambdat-%g_latent-1.mat'],trial,Para.lambda0,Para.lambdat);
        if exist(perf_filepath,'file')
            
            %%% Load Result
            load(perf_filepath,'meanAcc','Acc');
            fprintf('%dth trial acc = %.2f\n',trial,meanAcc(trial)*100);
            TrialAcc(trial) = meanAcc(trial);
        else
            continue;
        end
    
end

% fprintf('')