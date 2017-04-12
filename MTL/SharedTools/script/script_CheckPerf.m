%% sript to check performance
Dataset = 'OlympicSports'; % HMDB51, UCF101, OlympicSports, CCV
MathchingSpace = 'distributed'; % latent, distributed
EvalApproach = 'map'; % meanAcc, map
perf_path = sprintf('/import/geb-experiments-archive/Alex/MTL/%s/RMTL/Perf/',Dataset);

TrialAcc = zeros(1,50);
TrialMap = zeros(1,50);

for trial = 1:50
    
    switch MathchingSpace
        case 'latent'
            perf_filepath = sprintf([perf_path,'%s_trial-%d_lambda0-%g_lambdat-%g_latent-1.mat'],EvalApproach,trial,Para.lambda0,Para.lambdat);
        case 'distributed'
            perf_filepath = sprintf([perf_path,'%s_trial-%d_lambda0-%g_lambdat-%g_latent-0.mat'],EvalApproach,trial,Para.lambda0,Para.lambdat);
    end
    
    if exist(perf_filepath,'file')
        %%% Load Result
        switch EvalApproach
            case 'meanAcc'
                load(perf_filepath,'meanAcc','Acc');
                fprintf('%dth trial acc = %.2f\n',trial,meanAcc(trial)*100);
                TrialAcc(trial) = meanAcc(trial);
            case 'map'
                load(perf_filepath,'map','ap');
                fprintf('%dth trial map = %.2f\n',trial,map(trial)*100);
                TrialMap(trial) = map(trial);
        end
    else
        switch EvalApproach
            case 'meanAcc'
                fprintf('%dth trial acc = %.2f\n',trial,0);
            case 'map'
                fprintf('%dth trial map = %.2f\n',trial,0);
        end
        
        continue;
    end
    
end

switch EvalApproach
    case 'meanAcc'
        fprintf('%s meanAcc=%.2f stdAcc = %.2f\n',Dataset,100*mean(TrialAcc),100*std(TrialAcc));
    case 'map'
        fprintf('%s map=%.2f stdAcc = %.2f\n',Dataset,100*mean(TrialMap),100*std(TrialMap));
end




