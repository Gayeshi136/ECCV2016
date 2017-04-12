%% function for kernelized GO-MTL

function [D,A,S,Loss]=func_KernelizedMLE(K,Y,L,lambdaD,lambdaA,lambdaS)
% addpath(genpath('/import/geb-experiments/Alex/Matlab/Techniques/L1GeneralExamples'));
addpath('/import/geb-experiments/Alex/Matlab/Techniques/ADMM/function/');
% addpath('/import/geb-experiments/Alex/Matlab/Techniques/spams-matlab/build');

% if isempty(Term_KKt)
% end

%% Parameters
param.MaxItr = 20;
param.epsilon = 1e-3;

N_K = size(K,1);
N_Y = size(Y,1);
N_L = L;
Data.K = K;
Data.N_K = N_K;
Data.Y = Y;
Data.lambdaD = lambdaD;
Data.lambdaA = lambdaA;
Data.lambdaS = lambdaS;

%% Random Initialize
D = randn(N_K,N_L);
S = randn(N_L,N_K);
A = randn(N_L,N_Y);

%% Precompute Terms
Term_YYt = Y*Y';

itr=1;
Loss(itr) = func_Loss_MLE(D,A,S,Data);
fprintf('%d-th itr Loss=%g\n',itr,Loss(end));
itr=itr+1;

while true
    
    %% Fix S Update D
    D = K*S'/(S*S'+lambdaD*N_K*eye(N_L));
    
    %% Fix S Update A
    A = S*Y'/(Term_YYt + lambdaA*N_K*eye(N_Y));
    
    %% Fix D,A Update S
    S = (D'*D+(lambdaS+1)*N_K*eye(N_L))\(D'*K+A*Y);
    
    Loss(itr) = func_Loss_MLE(D,A,S,Data);
    fprintf('%d-th itr Loss=%g\n',itr,Loss(end));
    
    %% Check Convergence
    if abs(Loss(itr)-Loss(itr-1))/abs(Loss(itr)) < param.epsilon || itr >= param.MaxItr
        break;
    end
    
    itr=itr+1;
end

% %% Initialize Regressor by STL
% A0 = Y/(K+lambdaA*N_K*eye(N_K));
%
% [U,~,~] = svd(A0');
% A = U(:,1:L)';
% S = double(randn(N_Y,L));
% % S_admm = double(randn(N_T,L));
%
% %% Iterative Update A and S
% param.mode=2;
% param.lambda = lambdaS;
% param.lambda2 = 0;
% param.epsilon_full=5e-4;
% param.MaxItr = 8;
% itr = 1;
%
% Data.S = S;
% Data.Term_KKt = K*K';
% Data.Term_SAK = S*A*K;
% Data.Term_StYK = S'*Y*K;
% Data.Term_StS = S'*S;
%
% Data.Trace_YtY = sum(sum(Y.*Y));
% Data.K = K;
% Data.N_K = N_K;
% Data.Y = Y;
% Data.lambdaD = lambdaD;
% Data.lambdaA = lambdaA;
% Data.lambdaS = lambdaS;
%
% [Loss_full(itr),Grad_vec] = func_Loss(reshape(A,numel(A),1),Data);
% fprintf('Initial %d itr Loss=%g\n',itr,Loss_full(end));
%
% itr = itr+1;
%
% % funObj = @(s)SquaredError;
% % options.verbose = false;
%
% % param.lambda=2*lambdaS*N_K;
% % param.lambda2 = 0;
% % param.mode=2;
%
% while true
%
%     %% L1 Regularized Opt aka LASSO regression
%     % Update S
%     term_KtAt = K'*A';
%     tic;
%     parfor t = 1:N_Y
%         temp_S = lasso_admm(term_KtAt,Y(t,:)',2*N_K*lambdaS, 1.0, 1.0);
%         S(t,:) = temp_S';
%     end
%     toc;
%
%     Data.S = S;
%     Data.Term_KKt = K*K';
%     Data.Term_SAK = S*A*K;
%     Data.Term_StYK = S'*Y*K;
%     Data.Term_StS = S'*S;
%
%     [Loss_S(itr),Grad_vec] = func_Loss(reshape(A,numel(A),1),Data);
%     fprintf('Update S %d itr Loss=%g\n',itr,Loss_S(end));
% %     Data.S = S;
% %     Data.Term_SAK = S*A*K;
% %     Data.Term_StYK = S'*Y*K;
% %     Data.Term_StS = S'*S;
%     %% Gradient Descent to Solve A
%     % Update A
%     script_GonjugateDescent;
%
%     [Loss_A(itr),Grad_vec] = func_Loss(reshape(A,numel(A),1),Data);
%     fprintf('Update A %d itr Loss=%g\n',itr,Loss_A(end));
%     Loss_full(itr) = Loss_A(itr);
%     %% Check Convergence
%     if abs(Loss_full(itr)-Loss_full(itr-1))/abs(Loss_full(itr)) < param.epsilon_full || itr >= param.MaxItr
%         break;
%     end
%     itr = itr+1;
%
% end


