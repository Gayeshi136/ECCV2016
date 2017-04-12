%% function for kernelized GO-MTL

function [D,A,S,Loss]=func_KernelizedVideoStory(K,Y,L,lambdaD,lambdaA,lambdaS,param)
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
N_Y = size(Y,1);
N_L = L;
Data.K = K;
Data.N_K = N_K;
Data.Y = Y;
Data.lambdaD = lambdaD;
Data.lambdaA = lambdaA;
Data.lambdaS = lambdaS;

%% Random Initialize
D = randn(N_Y,N_L);
S = randn(N_L,N_K);
A = randn(N_L,N_K);

%% Precompute Terms
% Term_KKt = K*K';
Term_pinvKKt = K'/(K*K' + lambdaA*N_K*eye(N_K));

itr=1;
Loss(itr) = func_Loss_VideoStory(D,A,S,Data);
fprintf('%d-th itr Loss=%g\n',itr,Loss(end));
itr=itr+1;
tic;
while true
    
    %% Fix S Update D
    D = Y*S'/(S*S'+lambdaD*N_K*eye(N_L));
    
    %% Fix S Update A
    A = S*Term_pinvKKt;
    
    %% Fix D,A Update S
    S = (D'*D+(lambdaS*N_K+1)*eye(N_L))\(D'*Y+A*K);
    
    Loss(itr) = func_Loss_VideoStory(D,A,S,Data);
    fprintf('%d-th itr Loss=%g\n',itr,Loss(end));
    
    %% Check Convergence
    if abs(Loss(itr)-Loss(itr-1))/abs(Loss(itr)) < param.epsilon || itr >= param.MaxItr
        break;
    end
    
    itr=itr+1;
end
toc;

