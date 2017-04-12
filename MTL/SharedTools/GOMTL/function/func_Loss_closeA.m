%% function to compute loss and gradient

function [Loss] = func_Loss_closeA(A_vec,Data)


A = reshape(A_vec,numel(A_vec)/Data.N_K,Data.N_K);

Term_AK = A*Data.K;
Term_SAK = Data.S*Term_AK;

% Grad = 1/(2*Data.N_K)*(-2*Data.Term_StYK + 2*Data.Term_StS*A*Data.Term_KKt) +...
%     2*Data.lambdaA*A*Data.K;
% Grad_vec = reshape(Grad,numel(Grad),1);
Loss = 1/(Data.N_K)*sum(sum((Data.Y-Term_SAK).*(Data.Y-Term_SAK))) +...
    Data.lambdaA*sum(sum(Term_AK.*Term_AK)) + Data.lambdaS*sum(abs(Data.S(:)));

% fprintf('Loss=%g GradNorm=%g\n',Loss,norm(Grad,'fro'));
