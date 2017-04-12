%% script to train GOMTL on VideoStory Dataset
clear;

addpath('/import/geb-experiments/Alex/ECCV16/code/MTL/SharedTools/VideoStory/function/');
addpath('/import/geb-experiments/Alex/ECCV16/code/MTL/SharedTools/GOMTL/function/');
%% Parameters

%% Load VS data
script_LoadVSData;

%% Train GOMTL

%%% Compute Kernel
K = double(X'*X);
Y = double(Y);
L = 1024;


Para.lambdaS = 1e-4;
Para.lambdaA = 1e-4;

[Model.S,Model.A,Model.L]=func_KernelizedGOMTL_admm_ncg(K,Y,L,Para.lambdaS,Para.lambdaA);

% Para.lambdaS = 1e-7;
% Para.lambdaA = 1e-3;
% Para.LatentProportion = 0.2;    % proportion of latent tasks

model_path = '/import/geb-experiments-archive/Alex/MTL/VideoStory/GOMTL/Model/';
if ~exist(model_path,'dir')
    mkdir(model_path);
end

perf_path = '/import/geb-experiments-archive/Alex/MTL/VideoStory/GOMTL/Perf/';
if ~exist(perf_path,'dir')
    mkdir(perf_path);
end

model_filepath = sprintf([model_path,'LinearRegress_lambdaS-%g_lambdaA-%g_Lat-%g.mat'],Para.lambdaS,Para.lambdaA,L);
if ~exist(model_filepath,'file')
    fid = fopen(model_filepath,'w');
    fclose(fid);
else
    fprintf('Exist %s\n',model_filepath);
%     continue;
end

save(model_filepath,'Model');


