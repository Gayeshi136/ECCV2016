%% function to compute loss and gradient

function [Loss,Grad_vec] = func_Loss(A_vec,Data)


A = reshape(A_vec,numel(A_vec)/Data.N_K,Data.N_K);

Term_SAK = Data.S*A*Data.K;

Grad = 1/(2*Data.N_K)*(-2*Data.Term_StYK + 2*Data.Term_StS*A*Data.Term_KKt) +...
    2*Data.lambdaA*A*Data.K;
Grad_vec = reshape(Grad,numel(Grad),1);
Loss = 1/(2*Data.N_K)*(Data.Trace_YtY - 2*sum(sum(Data.Y.*Term_SAK)) + sum(sum(Term_SAK.*Term_SAK))) +...
    Data.lambdaA*trace(A*Data.K*A') + Data.lambdaS*sum(abs(Data.S(:)));

% fprintf('Loss=%g GradNorm=%g\n',Loss,norm(Grad,'fro'));
