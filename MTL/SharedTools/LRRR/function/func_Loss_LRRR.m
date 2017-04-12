%% function to compute loss

function L=func_Loss_LRRR(Data,A,S,K)

Term_Z_SAK = Data.Z - S*A*K;

Term_SA = S*A;

L = 1/Data.N_K*(sum(sum(Term_Z_SAK.*Term_Z_SAK))) + Data.lambda*sum(sum(Term_SA.*Term_SA));