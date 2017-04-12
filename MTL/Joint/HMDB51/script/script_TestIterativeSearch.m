%% Iterative search

Para.lambdaD_range = [1e-3 1e-4 1e-5];
Para.lambdaA_range = [1e-3 1e-4 1e-5];
Para.lambdaS_range = [1e-3 1e-4 1e-5];
Para.LatentDim_range = 60;



best_D = 1e-4;
best_A = 1e-5;
best_S = 1e-3;
best_L = 60;

bestAcc = 0;
best_idx = 0;

for trial = 1:4
    
    for paraD = Para.lambdaD_range
        
        idx = find(Perf_Lat(:,2)==best_A & Perf_Lat(:,3)==best_S & Perf_Lat(:,4)==best_L);
        [tempAcc,temp_idx] = max(Perf_Lat(idx,5));
        temp_D = Perf_Lat(idx(temp_idx),1);
        
        if bestAcc<tempAcc
            best_D = temp_D;
            bestAcc = tempAcc;
            best_idx = idx(temp_idx);
        end
    end
    
    for A = Para.lambdaA_range
        
        idx = find(Perf_Lat(:,1)==best_D & Perf_Lat(:,3)==best_S & Perf_Lat(:,4)==best_L);
        [tempAcc,temp_idx] = max(Perf_Lat(idx,5));
        temp_A = Perf_Lat(idx(temp_idx),1);
        
        if bestAcc<tempAcc
            best_A = temp_A;
            bestAcc = tempAcc;
            best_idx = idx(temp_idx);
        end
    end
    
    for S = Para.lambdaS_range
        
        idx = find(Perf_Lat(:,1)==best_D & Perf_Lat(:,2)==best_A & Perf_Lat(:,4)==best_L);
        [tempAcc,temp_idx] = max(Perf_Lat(idx,5));
        temp_S = Perf_Lat(idx(temp_idx),1);
        
        if bestAcc<tempAcc
            best_S = temp_S;
            bestAcc = tempAcc;
            best_idx = idx(temp_idx);
        end
    end
    
    for L = Para.LatentDim_range
        
        idx = find(Perf_Lat(:,1)==best_D & Perf_Lat(:,2)==best_A & Perf_Lat(:,3)==best_S);
        [tempAcc,temp_idx] = max(Perf_Lat(idx,5));
        temp_L = Perf_Lat(idx(temp_idx),1);
        
        if bestAcc<tempAcc
            best_L = temp_L;
            bestAcc = tempAcc;
            best_idx = idx(temp_idx);
        end
        
    end
    bestAcc
    
end
