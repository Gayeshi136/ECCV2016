%% function for kernelized GO-MTL

function [S,A,Loss_full]=func_KernelizedL2Classification(K,Y,lambda,gamma)
% addpath(genpath('/import/geb-experiments/Alex/Matlab/Techniques/L1GeneralExamples'));
addpath('/import/geb-experiments/Alex/Matlab/Techniques/ADMM/function/');
addpath('/import/geb-experiments/Alex/Matlab/Techniques/poblano_toolbox');

% addpath('/import/geb-experiments/Alex/Matlab/Techniques/spams-matlab/build');

% if isempty(Term_KKt)
% end

N_K = size(K,1);
N_T = size(Y,1);
N_L = L;

%% Solve L2 Classification Loss



%% Initialize Regressor by STL
A0 = Y/(K+lambdaA*N_K*eye(N_K));

[U,~,~] = svd(A0');
A = U(:,1:L)';
S = double(randn(N_T,L));
% S_admm = double(randn(N_T,L));

%% Iterative Update A and S
param.mode=2;
param.lambda = lambdaS;
param.lambda2 = 0;
param.epsilon_full=5e-4;
param.MaxItr = 10;
itr = 1;

Data.S = S;
Data.Term_KKt = K*K';
Data.Term_SAK = S*A*K;
Data.Term_StYK = S'*Y*K;
Data.Term_StS = S'*S;

Data.Trace_YtY = sum(sum(Y.*Y));
Data.K = K;
Data.N_K = N_K;
Data.Y = Y;
Data.lambdaA = lambdaA;
Data.lambdaS = lambdaS;

[Loss_full(itr),Grad_vec] = func_Loss(reshape(A,numel(A),1),Data);
fprintf('Initial %d itr Loss=%g\n',itr,Loss_full(end));

itr = itr+1;

% funObj = @(s)SquaredError;
% options.verbose = false;

% param.lambda=2*lambdaS*N_K;
% param.lambda2 = 0;
% param.mode=2;

while true
    
    %% L1 Regularized Opt aka LASSO regression
    % Update S
    term_KtAt = K'*A';
    tic;
    parfor t = 1:N_T
        temp_S = lasso_admm(term_KtAt,Y(t,:)',2*N_K*lambdaS, 1.0, 1.0);
        fprintf('finish task=%d\n',t);
        S(t,:) = temp_S';
    end
    toc;
    
    Data.S = S;
    Data.Term_KKt = K*K';
    Data.Term_SAK = S*A*K;
    Data.Term_StYK = S'*Y*K;
    Data.Term_StS = S'*S;
    
    [Loss_S(itr),Grad_vec] = func_Loss(reshape(A,numel(A),1),Data);
    fprintf('Update S %d itr Loss=%g\n',itr,Loss_S(end));
%     Data.S = S;
%     Data.Term_SAK = S*A*K;
%     Data.Term_StYK = S'*Y*K;
%     Data.Term_StS = S'*S;
    %% Gradient Descent to Solve A
    % Update A
    script_GonjugateDescent;
    
    [Loss_A(itr),Grad_vec] = func_Loss(reshape(A,numel(A),1),Data);
    fprintf('Update A %d itr Loss=%g\n',itr,Loss_A(end));
    Loss_full(itr) = Loss_A(itr);
    %% Check Convergence
    if abs(Loss_full(itr)-Loss_full(itr-1))/abs(Loss_full(itr)) < param.epsilon_full || itr >= param.MaxItr
        break;
    end
    itr = itr+1;
    
end


