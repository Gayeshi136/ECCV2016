%% function for kernelized GO-MTL

function [S,A,L]=func_tr_KernelizedL2Classification_MultiTask(K,V,Y,A0,Para)
% addpath(genpath('/import/geb-experiments/Alex/Matlab/Techniques/L1GeneralExamples'));
addpath('/import/geb-experiments/Alex/Matlab/Techniques/ADMM/function/');
addpath('/import/geb-experiments/Alex/Matlab/Techniques/poblano_toolbox');

% addpath('/import/geb-experiments/Alex/Matlab/Techniques/spams-matlab/build');

% if isempty(Term_KKt)
% end


N = size(K,1);
t = Para.latent;
Dv = size(V,1);
lambda = Para.lambda;
gamma = Para.gamma;
error = 1e-5;
% lambda = 1e-3;
% gamma = 1e-2;
%% Solve L2 Classification Loss
pinv_K = pinv(K);
pinv_V = pinv(V);

%% Initialize A and S
if ~exist('A0','var')
    Para.gammaA0 = 1e-3;
    Para.lambdaA0 = 1e-2;
    term_VYK = V*Y*K;
    term_VVt = V*V'+Para.gammaA0*N*eye(size(V,1));
    term_KKt = K*K'+Para.lambdaA0*N*eye(N);
    A0 = inv(term_VVt)*term_VYK*inv(term_KKt);
end

[U,sig,Ve] = svd(A0);
Vet = Ve';
S = U(:,1:t)';
A = sig(1:t,1:t)*Vet(1:t,:);

%% Calculate Initial Error
term_VVt = V*V';
term_SV = S*V;
term_AK = A*K;
L = 0;
L=L2ClassificationLoss(term_SV,term_AK,Y,lambda,gamma,N);

%% Iteratively Update A and S
itr = 2;
while 1
    S = (term_AK*term_AK'+gamma*N*eye(t))\(term_AK*Y'*pinv_V);
    A = (term_SV*term_SV'+lambda*N*eye(t))\(term_SV*Y*pinv_K);
    %     S = (term_AK*term_AK'+gamma*N*eye(t))\(term_AK*Y'*pinv_V);
    
    term_SV = S*V;
    term_AK = A*K;
    
    L(itr)=L2ClassificationLoss(term_SV,term_AK,Y,lambda,gamma,N);
    fprintf('itr--%d,  L=%.4f\n',itr,L(itr));
    itr = itr+1;
    
    %% Judge Exit
    if L(end-1)-L(end)<error
        break;
    end
end


function L=L2ClassificationLoss(term_SV,term_AK,Y,lambda,gamma,N)

loss_term = term_SV'*term_AK-Y;

L = 1/N*sum(sum(loss_term.*loss_term))+lambda*sum(sum(term_AK.*term_AK)) + gamma*sum(sum(term_SV.*term_SV));

% n_l = size(K,1);
% term_VYK = V*Y*K;
% term_VVt = V*V'+Para.gamma*n_l*eye(size(V,1));
% term_KKt = K*K'+Para.lambda*n_l*eye(n_l);
% A = inv(term_VVt)*term_VYK*inv(term_KKt);

% %% Initialize Regressor by STL
% A0 = Y/(K+lambdaA*N_K*eye(N_K));
%
% [U,~,~] = svd(A0');
% A = U(:,1:L)';
% S = double(randn(N_T,L));
% % S_admm = double(randn(N_T,L));
%
% %% Iterative Update A and S
% param.mode=2;
% param.lambda = lambdaS;
% param.lambda2 = 0;
% param.epsilon_full=5e-4;
% param.MaxItr = 10;
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
%     parfor t = 1:N_T
%         temp_S = lasso_admm(term_KtAt,Y(t,:)',2*N_K*lambdaS, 1.0, 1.0);
%         fprintf('finish task=%d\n',t);
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


