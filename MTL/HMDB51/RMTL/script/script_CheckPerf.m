%% sript to check performance
perf_path = '/import/geb-experiments-archive/Alex/MTL/HMDB51/RMTL/Perf/';

TrialAcc = zeros(1,50);

for trial = 1:50
   
    perf_filepath = sprintf([perf_path,'meanAcc_trial-%d_lambda0-%g_lambdat-%g.mat'],trial,Para.lambda0,Para.lambdat);
        if exist(perf_filepath,'file')
            
            %%% Load Result
            load(perf_filepath,'meanAcc','Acc');
            fprintf('%dth trial acc = %.2f\n',trial,meanAcc(trial)*100);
            TrialAcc(trial) = meanAcc(trial);
        else
            continue;
        end
    
end