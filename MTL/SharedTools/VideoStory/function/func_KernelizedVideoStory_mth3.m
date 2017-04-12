%% function for kernelized GO-MTL

function [D,A,S,Loss]=func_KernelizedVideoStory_mth3(K,Z,L,weight,lambdaD,lambdaA,lambdaS,param)
% addpath(genpath('/import/geb-experiments/Alex/Matlab/Techniques/L1GeneralExamples'));
addpath('/import/geb-experiments/Alex/Matlab/Techniques/ADMM/function/');
% addpath('/import/geb-experiments/Alex/Matlab/Techniques/spams-matlab/build');

% if isempty(Term_KKt)
% end

%% Parameters
if ~exist('param','var')
    param.MaxItr = 100;
    param.epsilon = 1e-4;
end

N_K = size(K,1);
N_Y = size(Z,1);
N_L = L;
Para.K = K;
Para.N_K = N_K;
Para.lambdaD = lambdaD;
Para.lambdaA = lambdaA;
Para.lambdaS = lambdaS;
W = sparse(1:N_K,1:N_K,sqrt(weight)); % diagonal matrix of root square weight
% Term_WWt = W*W';

%% Preprocess weighted word-vector and kernel
K_weight = K*W;
% Z_weight = Z*W;

%% Random Initialize
D = randn(N_Y,N_L);
S = randn(N_L,N_K);
% S_weight = S*W;
% A = randn(N_L,N_K);

%% Precompute Terms
% Term_KKt = K*K';

itr=1;
Loss(itr) = func_Loss_VideoStory_weight_mth3(D,S,Z,Para);
fprintf('%d-th itr Loss=%g\n',itr,Loss(end));
itr=itr+1;

while true
    
    %% Fix S Update D
    D = Z*S'/(S*S'+lambdaD*N_K*eye(N_L));

    %% Fix D,A Update S
    S = (D'*D+(lambdaS*N_K+1)*eye(N_L))\(D'*Z);
%     S_weight = S*W;
    
    Loss(itr) = func_Loss_VideoStory_weight_mth3(D,S,Z,Para);
    fprintf('%d-th itr Loss=%g\n',itr,Loss(end));
    
    %% Check Convergence
    if abs(Loss(itr)-Loss(itr-1))/abs(Loss(itr)) < param.epsilon || itr >= param.MaxItr
        break;
    end
    
    itr=itr+1;
end

Term_pinvKKt = K_weight'/(K_weight*K_weight' + lambdaA*N_K*eye(N_K));
S_weight = S*W;
%% Fix S Update A
A = S_weight*Term_pinvKKt;
