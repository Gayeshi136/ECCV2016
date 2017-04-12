%% function for kernelized GO-MTL

function [A]=func_tr_KernelizedHingeClassification(K,V,Y,Para)
% addpath(genpath('/import/geb-experiments/Alex/Matlab/Techniques/L1GeneralExamples'));
addpath('/import/geb-experiments/Alex/Matlab/Techniques/ADMM/function/');
addpath('/import/geb-experiments/Alex/Matlab/Techniques/poblano_toolbox');

% addpath('/import/geb-experiments/Alex/Matlab/Techniques/spams-matlab/build');

% if isempty(Term_KKt)
% end
if min(Y(:))~=-1
    Y = 2*(Y-0.5);
end

N = size(K,1);
m = size(Y,1);
N_T = size(Y,1);
Dv = size(V,1);
lambda = Para.lambda;
gamma = Para.gamma;
C = Para.C;
% term_VYK = V*Y*K;
term_VVt = V*V'+Para.gamma*N*eye(size(V,1));
term_KKt = K*K'+Para.lambda*N*eye(N);
alpha = 0.01;
%% Random Sample Training Samples
indx = randsample(1:N,3*N,true);

%% Solve Hinge Loss
A = randn(Dv,N);
violation = 0;
violation_itr = [];
batchSize = 100;
batch = 1;
figure; hold on;

for itr = 1:numel(indx)
    tr_i = indx(itr);
    for m_i = 1:m
       loss_term = Y(m_i,tr_i)'*V(:,m_i)'*A*K(:,tr_i);
        if loss_term< C
           violation = violation+1;
            A = A- alpha*(-V(:,m_i)*Y(m_i,tr_i)*K(:,tr_i)' + 2*lambda*A*term_KKt + 2*gamma*term_VVt*A + 2*gamma*lambda*A);
            
        end
        
    end   
    
    if mod(itr,batchSize)
        fprintf('Vilation=%d\n',violation);
        violation_itr(batch) = violation;
        if numel(violation_itr) < 2
            plot(batch,violation);hold on;
        else
            plot([batch-1, batch],[violation_itr(end-1:end)]);hold on;
        end
        batch = batch+1;
        drawnow;
        violation = 0;
    end
end

% n_l = size(K,1);
% term_VYK = V*Y*K;
% term_VVt = V*V'+Para.gamma*n_l*eye(size(V,1));
% term_KKt = K*K'+Para.lambda*n_l*eye(n_l);
% A = inv(term_VVt)*term_VYK*inv(term_KKt);
% 
% A=zeros(size(A));
% term_AK = A*K;
% term_VtA = V'*A;
% loss_term = V'*A*K-Y;
% 
% L = 1/n_l*sum(sum(loss_term.*loss_term))+gamma*sum(sum(term_AK.*term_AK)) + lambda*sum(sum(term_VtA.*term_VtA)) + ...
%     lambda*gamma*sum(sum(A.*A));


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


