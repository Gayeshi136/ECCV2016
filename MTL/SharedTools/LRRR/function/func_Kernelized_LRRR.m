%% function for kernelized GO-MTL

function [S,A,Loss]=func_Kernelized_LRRR(K,Z,L,lambda)
% addpath(genpath('/import/geb-experiments/Alex/Matlab/Techniques/L1GeneralExamples'));
% addpath('/import/geb-experiments/Alex/Matlab/Techniques/ADMM/function/');
% addpath('/import/geb-experiments/Alex/Matlab/Techniques/spams-matlab/build');

% if isempty(Term_KKt)
% end

N_K = size(K,1);
N_Z = size(Z,1);
N_L = L;

%% Initialize Regressor by STL
% A0 = Z/(K+lambdaA*N_K*eye(N_K));
% 
% [U,~,~] = svd(A0');
% A = U(:,1:L)';
% S = double(randn(N_Z,L));
% S_admm = double(randn(N_T,L));

%% Random Initialize
A = randn(N_L,N_K);
S = randn(N_Z,N_L);

%% Iterative Update A and S
param.mode=2;
param.lambda = lambda;
param.epsilon=5e-4;
param.MaxItr = 10;
itr = 1;

% Data.S = S;
Data.Term_KKt = K*K';
% Data.Term_SAK = S*A*K;
% Data.Term_StYK = S'*Z*K;
Data.Term_ZKt = Z*K';
% Data.Term_invStS = inv(S'*S);
Data.K = K;
Data.N_K = N_K;
Data.Z = Z;
Data.lambda = lambda;

[Loss(itr)] = func_Loss_LRRR(Data,A,S,K);
fprintf('Initial %d itr Loss=%g\n',itr,Loss(end));

itr = itr+1;

while true
    
    %% update A
    A = inv(S'*S)*(S'*Data.Term_ZKt)*inv(Data.Term_KKt+Data.lambda*N_K*eye(N_K));
    
    %% Update S
    S = (Data.Term_ZKt * A')/(A*Data.Term_KKt*A' + Data.lambda*N_K*A*A');


    Loss(itr)=func_Loss_LRRR(Data,A,S,K);

    fprintf('Update A %d itr Loss=%g\n',itr,Loss(end));
    %% Check Convergence
    if abs(Loss(itr)-Loss(itr-1))/abs(Loss(itr)) < param.epsilon || itr >= param.MaxItr
        break;
    end
    itr = itr+1;
    
end


