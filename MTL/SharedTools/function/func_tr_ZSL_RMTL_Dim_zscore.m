%% Regularized Multi-Task Learning to train regression
%
%   each dimension in word vector is treated as a single task

function ReturnVar=func_tr_ZSL_RMTL_Dim_zscore(Para)

global D

addpath('/import/geb-experiments/Alex/CVPR15/Code/Zeroshot/HMDBregression/Code/Basic/FVNormalization');
addpath('/import/geb-experiments/Alex/ICCV15/code/TransferRegression/CollectData/function');
addpath('/import/geb-experiments/Alex/ICCV15/code/SharedTools/function/');

%% parameters
Para.N_itr_L = 10;
Para.StopError = 1e-4;

%% Generate Training Kernel Matrix

Z = (Para.Z)';
Z_sumt = sum(Z,1);
T = size(Z,1); % number of tasks
N = size(Z,2); % number of instances

K = D(Para.aug_tr_idx,Para.aug_tr_idx);


%% Iterative Optimize At and A0
L = inf;
A0 = [];
At_bar = randn(size(Z,1),size(K,2));

itr = 1;

%%% Precompute Inversion Term
InversionTerm_A0 = inv(T*K + Para.lambda0*N*eye(size(K)));
InversionTerm_At = inv(K + Para.lambdat*N/T*eye(size(K)));
% Z_InvTerm_A0 = Z_sumt*InversionTerm_A0;
Z_InvTerm_At = Z*InversionTerm_At;
K_InvTerm_At = K*InversionTerm_At;
tic;
while true
    A0 = (Z_sumt - sum(At_bar*K,1))*InversionTerm_A0;
    A0_bar = repmat(A0,size(At_bar,1),1);
    
%     At_bar = (Z - A0_bar*K)*InversionTerm_At;
    At_bar = Z_InvTerm_At - A0_bar*K_InvTerm_At;
    %% Compute Loss Function
    if ~mod(itr,Para.N_itr_L) || itr == 1
%         term1 = (A0_bar+At_bar)*K;
%         L_temp = 1/N*(trace(-2*Z'*term1 + term1'*term1)) +...
%             Para.lambdat/T*trace(At_bar*K*At_bar')+Para.lambda0*A0*K*A0';
        %%% Fast Implementaion of trace of matrices multiplication
        term1 = (A0_bar+At_bar)*K;
        L_temp = 1/N*(-2*sum(sum(Z.*term1))+...
            sum(sum(term1.*term1)))+...
            Para.lambdat/T*trace(At_bar*K*At_bar')+Para.lambda0*A0*K*A0';
        L = [L L_temp];
    end
    %     if itr>200
    %         break;
    %     end
    
    %% Check Convergence
    if ~mod(itr,Para.N_itr_L) && abs((L(end)-L(end-1))/L(end))<Para.StopError
        break;
    end
        
    t_elapse = toc;
    if ~mod(itr,Para.N_itr_L) || itr == 1
        fprintf('Update At itr=%d L=%g in %d seconds\n',itr,L(end),t_elapse);
    else
        fprintf('Update At itr=%d in %d seconds\n',itr,t_elapse);
    end
    tic;
    
    itr = itr+1;
end

%% Compute Loss at the end
term1 = (A0_bar+At_bar)*K;
L_temp = 1/N*(-2*sum(sum(Z.*term1))+...
    sum(sum(term1.*term1)))+...
    Para.lambdat/T*trace(At_bar*K*At_bar')+Para.lambda0*A0*K*A0';
L = [L L_temp];

t_final_elapse = toc;
fprintf('Final itr=%d L=%g in %d seconds\n',itr,L(end),t_final_elapse);

ReturnVar.A0 = A0;
ReturnVar.At_bar = At_bar;
ReturnVar.L = L;
ReturnVar.itr = itr;



