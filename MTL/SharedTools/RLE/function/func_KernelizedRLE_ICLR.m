%% function for kernelized GO-MTL

function [S,A,Loss_full]=func_KernelizedRLE_ICLR(K,Y,L,lambdaS,lambdaA)
% addpath(genpath('/import/geb-experiments/Alex/Matlab/Techniques/L1GeneralExamples'));
addpath('/import/geb-experiments/Alex/Matlab/Techniques/ADMM/function/');
% addpath('/import/geb-experiments/Alex/Matlab/Techniques/spams-matlab/build');

% if isempty(Term_KKt)
% end

N_K = size(K,1);
N_D = size(Y,1);
N_L = L;

%% Initialize Regressor by STL
A0 = Y/(K+2*lambdaA*N_K*eye(N_K));

[U,~,~] = svd(A0');
A = U(:,1:L)';
S = double(randn(L,N_D));
% S_admm = double(randn(N_T,L));

%% Iterative Update A and S
param.mode=2;
param.lambda = lambdaS;
param.lambda2 = 0;
param.epsilon_full=5e-4;
param.MaxItr = 15;
itr = 1;

Data.S = S;
Data.Term_SY = S*Y;

Data.Trace_YtY = sum(sum(Y.*Y));
Data.K = K;
Data.N_K = N_K;
Data.Y = Y;
Data.lambdaA = lambdaA;
Data.lambdaS = lambdaS;

[Loss_full(itr)] = func_Loss_ICLR(A,Data);
fprintf('Initial %d itr Loss=%g\n',itr,Loss_full(end));

itr = itr+1;

while true
    
    S = 1/(2*lambdaS)*A*K*Y';
    Data.S = S;

    A = 1/(2*lambdaA)*S*Y;
    
    [Loss(itr)] = func_Loss(A,Data);
    fprintf('Update S & A %d itr Loss=%g\n',itr,Loss(end));
    itr = itr+1;
    %% L1 Regularized Opt aka LASSO regression
    % Update S
%     term_KtAt = K'*A';
%     tic;
%     parfor l = 1:N_L
%         temp_S = lasso_admm(Y',term_KtAt(:,l),2*N_K*lambdaS, 1.0, 1.0);
%         S(l,:) = temp_S';
%     end
%     toc;
%     
%     Data.S = S;
%     Data.Term_SY = S*Y;
%     
%     [Loss_S(itr)] = func_Loss(A,Data);
%     fprintf('Update S %d itr Loss=%g\n',itr,Loss_S(end));
% %     Data.S = S;
% %     Data.Term_SAK = S*A*K;
% %     Data.Term_StYK = S'*Y*K;
% %     Data.Term_StS = S'*S;
%     %% Gradient Descent to Solve A
%     % Update A
%     A = Data.Term_SY/(K+2*lambdaA*Data.N_K*eye(Data.N_K));
%     
%     [Loss_A(itr)] = func_Loss(A,Data);
%     fprintf('Update A %d itr Loss=%g\n',itr,Loss_A(end));
%     Loss_full(itr) = Loss_A(itr);
%     %% Check Convergence
%     if abs(Loss_full(itr)-Loss_full(itr-1))/abs(Loss_full(itr)) < param.epsilon_full || itr >= param.MaxItr
%         break;
%     end
%     itr = itr+1;
    
end


