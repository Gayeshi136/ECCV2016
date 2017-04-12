%% sript to check performance
perf_path = '/import/geb-experiments-archive/Alex/MTL/OlympicSports/RMTL/Perf/';
Para.lambda0 = 1e-2;
Para.lambdat = 1e-1;

TrialAcc = zeros(1,50);

for trial = 1:50
    
    perf_filepath = sprintf([perf_path,'map_trial-%d_lambda0-%g_lambdat-%g_latent-0.mat'],trial,Para.lambda0,Para.lambdat);
    if exist(perf_filepath,'file')
        
        %%% Load Result
        load(perf_filepath,'map','ap');
        fprintf('%dth trial acc = %.2f\n',trial,map(trial)*100);
        TrialMap(trial) = map(trial);
    else
        continue;
    end
    
end

mean(map)
std(map)