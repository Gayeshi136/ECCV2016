
A0=A;
param_ncg = ncg('defaults');
param_ncg.Update = 'PR';
param_ncg.Display = 'iter';
param_ncg.RelFuncTol = 5e-4;
param_ncg.StopTol = 1e-6/N_K;
param_ncg.MaxIters = 50;
A0_vec = reshape(A0,numel(A0),1);
out = ncg(@(x)func_Loss(x,Data),A0_vec,param_ncg);
A = reshape(out.X,size(A0,1),size(A0,2));
