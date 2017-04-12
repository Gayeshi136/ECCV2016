%% function for kernelized GO-MTL

function [S,A,Loss]=func_KernelizedGOMTL_L2_ncg(K,Y,L,lambdaS,lambdaA)
% addpath(genpath('/import/geb-experiments/Alex/Matlab/Techniques/L1GeneralExamples'));
addpath('/import/geb-experiments/Alex/Matlab/Techniques/ADMM/function/');
addpath('/import/geb-experiments/Alex/Matlab/Techniques/poblano_toolbox');

% addpath('/import/geb-experiments/Alex/Matlab/Techniques/spams-matlab/build');

% if isempty(Term_KKt)
% end

if ~exist('param','var')
    param.epsilon_full=1e-3;
    param.MaxItr = 50;
end

N_K = size(K,1);
N_T = size(Y,1);
N_L = L;

%% Initialize Regressor by STL
A0 = Y/(K+lambdaA*N_K*eye(N_K));
[U,sig,Ve] = svd(A0);
Vet = Ve';
S = U(:,1:N_L);
A = sig(1:N_L,1:N_L)*Vet(1:N_L,:);

% %% Initialize Regressor by STL
% A0 = Y/(K+lambdaA*N_K*eye(N_K));
% 
% [U,~,~] = svd(A0');
% A = U(:,1:L)';
% S = double(randn(N_T,L));
% % S_admm = double(randn(N_T,L));

%% Iterative Update A and S
param.mode=2;
param.lambda = lambdaS;
param.lambda2 = 0;

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

[Loss(itr)] = func_Loss_GOMTL_L2(A,S,Data);
fprintf('Initial %d itr Loss=%g\n',itr,Loss(end));

itr = itr+1;

% funObj = @(s)SquaredError;
% options.verbose = false;

% param.lambda=2*lambdaS*N_K;
% param.lambda2 = 0;
% param.mode=2;
Term_YKt = Y*K';
Term_KKt = K*K';

alpha_S = 1e-1;
alpha_A = 0.05;

while true
    
    %% L1 Regularized Opt aka LASSO regression
    % Update S
    S = (Term_YKt*A')/(A*Term_KKt*A'+lambdaS*N_K*eye(N_L));
    %     GradS = 1/N_K*(-2*Term_YKt*A'+2*S*A*Term_KKt*A')+2*lambdaS*S;
    %     while true
    %         S_attempt = S-alpha_S * GradS;
    %         Loss_attempt = func_Loss_GOMTL_L2(A,S_attempt,Data);
    %         if Loss_attempt < Loss(end)
    %             S=S_attempt;
    %             break;
    %         else
    %             alpha_S = 0.5*alpha_S;
    %             continue;
    %
    %         end
    %     end
    
    % Update A
    GradA = 1/N_K*(-2*S'*Term_YKt + 2*S'*S*A*Term_KKt) + 2*lambdaA*A;
    while true
        
        A_attempt = A-alpha_A * GradA;
        Loss_attempt = func_Loss_GOMTL_L2(A_attempt,S,Data);
        if Loss_attempt < Loss(end)
            A=A_attempt;
            break;
        else
            alpha_A = 0.5*alpha_A;
            continue;
            
        end
    end
    
    Loss(itr) = func_Loss_GOMTL_L2(A,S,Data);
    
    fprintf('Update A %d itr Loss=%g\n',itr,Loss(end));
    
    %% Check Convergence
    if abs(Loss(itr)-Loss(itr-1))/abs(Loss(itr)) < param.epsilon_full || itr >= param.MaxItr
        break;
    end
    itr = itr+1;
    
end


