%% function for kernelized GO-MTL

function [S,A,Loss_full]=func_KernelizedGOMTL_admm_close(K,Y,L,lambdaS,lambdaA,param)
% addpath(genpath('/import/geb-experiments/Alex/Matlab/Techniques/L1GeneralExamples'));
addpath('/import/geb-experiments/Alex/Matlab/Techniques/ADMM/function/');
addpath('/import/geb-experiments/Alex/Matlab/Techniques/poblano_toolbox');

% addpath('/import/geb-experiments/Alex/Matlab/Techniques/spams-matlab/build');

% if isempty(Term_KKt)
% end

if ~exist('param','var')
    param.epsilon_full=1e-3;
    param.MaxItr = 10;
end

N_K = size(K,1);
N_T = size(Y,1);
N_L = L;

%% Initialize Regressor by STL
A0 = Y/(K+lambdaA*N_K*eye(N_K));
%
% [U,~,~] = svd(A0');
% A = U(:,1:L)';
% S = double(randn(N_T,L));


[U,sig,Ve] = svd(A0);
Vet = Ve';
S = U(:,1:N_L);
A = sig(1:N_L,1:N_L)*Vet(1:N_L,:);

% S_admm = double(randn(N_T,L));

%% Iterative Update A and S
param.mode=2;
param.lambda = lambdaS;
param.lambda2 = 0;

itr = 1;

Data.S = S;
% Data.Term_KKt = K*K';
Data.pinvK = pinv(K);
% Data.Term_SAK = S*A*K;
Data.Term_StY = S'*Y;
Data.Term_StS = S'*S;

Data.Trace_YtY = sum(sum(Y.*Y));
Data.K = K;
Data.N_K = N_K;
Data.Y = Y;
Data.lambdaA = lambdaA;
Data.lambdaS = lambdaS;

Loss_full = [];

[Loss_full(itr)] = func_Loss_closeA(reshape(A,numel(A),1),Data);
fprintf('Initial %d itr Loss=%g\n',itr,Loss_full(end));

itr = itr+1;

% funObj = @(s)SquaredError;
% options.verbose = false;

% param.lambda=2*lambdaS*N_K;
% param.lambda2 = 0;
% param.mode=2;

Loss_A = [];
Loss_S = [];

while true
    
    %% L1 Regularized Opt aka LASSO regression
    % Update S
    term_KtAt = K'*A';
    tic;
    parfor t = 1:N_T
        temp_S = lasso_admm(term_KtAt,Y(t,:)',N_K/2*lambdaS, 1.0, 1.0);
        fprintf('finish task=%d\n',t);
        S(t,:) = temp_S';
    end
    toc;
    
    Data.S = S;
    Data.Term_StY = S'*Y;
    Data.Term_StS = S'*S;
    
    [Loss_S(itr)] = func_Loss_closeA(reshape(A,numel(A),1),Data);
    fprintf('Update S %d itr Loss=%g\n',itr,Loss_S(end));

    %% Gradient Descent to Solve A
    A = (Data.Term_StS + lambdaA*N_K*eye(N_L))\(Data.Term_StY*Data.pinvK);
    
    [Loss_A(itr)] = func_Loss_closeA(reshape(A,numel(A),1),Data);
    fprintf('Update A %d itr Loss=%g\n',itr,Loss_A(end));
    Loss_full(itr) = Loss_A(itr);
    %% Check Convergence
    if abs(Loss_full(itr)-Loss_full(itr-1))/abs(Loss_full(itr)) < param.epsilon_full || itr >= param.MaxItr
        break;
    end
    itr = itr+1;
    
end


