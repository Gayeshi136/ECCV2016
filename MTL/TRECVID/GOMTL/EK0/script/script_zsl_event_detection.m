clc;
clear all;

% add libsvm path for training event classifiers
addpath('/import/vision-datasets3/VideoStory/vs//libs/libsvm-3.17/matlab');

% load event train data - They should be l2 normalized
load('/import/vision-datasets3/VideoStory/vs//datasets/event_train/feature_mbh.mat');
trnData = data';

% load event train labels for 10Ex setting
load('/import/vision-datasets3/VideoStory/vs//datasets/event_train/labels10Ex.mat');
trnLabel = labels';

% load event test data - They should be l2 normalized
load('/import/vision-datasets3/VideoStory/vs//datasets/event_test/feature_mbh.mat');
tstData = data';

% load event test labels
load('/import/vision-datasets3/VideoStory/vs//datasets/event_test/labels.mat');
tstLabel = labels';

% augment the features with a bias parameter of 0.1 fixed in our
% experiments
% trnData = [trnData ; 0.1*ones(1, size(trnData, 2))];
% tstData = [tstData ; 0.1*ones(1, size(tstData, 2))];

% load the TRAINED video story mappings A and W for k = 1024, which is the
% optimal value as shown in the paper (Figure 5)
%% Load GOMTL model
K = 1024;
load(['/import/vision-datasets3/VideoStory/vs/code/AW_k' num2str(K) '.mat']);

% obtain the videostory represnetation (S) for event train and test data by
% projecting them through the visual mapping W
trnDataX = W' * trnData;
tstDataX = W' * tstData;

% l2 normalization of train data
for i = 1:size(trnData, 2)
    trnDataX(:, i) = trnDataX(:, i) ./ norm(trnDataX(:, i), 2);
end
trnDataX(isnan(trnDataX)) = 0;

% l2 normalization of test data
for i = 1:size(tstDataX, 2)
    tstDataX(:, i) = tstDataX(:, i) ./ norm(tstDataX(:, i), 2);
end
tstDataX(isnan(tstDataX)) = 0;

%% Load Prototypes
method = 'vsembedding';
trecvid_path = '/import/vision-datasets3/VideoStory/vs/datasets/TRECVID/';
vsembeding_path = sprintf('%s/VS_Embedding/',trecvid_path);

load(sprintf('%s/vsembedding_info-%s_m-%s.mat',vsembeding_path,'evtname',method),'S_ClassName','EvtList');

Prototypes = cell2mat(S_ClassName(EvtList))';

SS = sum(Prototypes.^2,2);
label_k = sqrt(size(Prototypes,2)./SS);
Prototypes = repmat(label_k,1,size(Prototypes,2)) .* Prototypes;

%% perform event detection
aps = zeros(1, size(trnLabel, 1));

% for each event
for e = 1:size(trnLabel, 1)
    
        % prepare the event testing data
    label_test = tstLabel(e, :);
    tstDataX2 = tstDataX(:, label_test ~= 0);
    label_test = label_test(label_test ~= 0);        
    
%     tslabel = tstLabel(e, :);
% %     D_ts_e = D_ts(tslabel ~=0 , trlabel ~=0);
%     label_test = tslabel(tslabel ~= 0);
% 
%     tstSentenceVector = trvd_tsSentenceVector(label_test ~= 0 , :);
    
    %% Do Knn to detect relevant event videos
    IDX = knnsearch(tstDataX2',Prototypes(e,:),'K',size(tstDataX2,2));
    aps(e) = func_calculate_map_zsl(IDX, label_test');
    
end

%% Save Results
save_path = '/import/vision-datasets3/VideoStory/vs/datasets/TRECVID/MED_EK0/VS_Embedding/results/';

if ~exist(save_path,'dir')
    mkdir(save_path);
end

save(sprintf('%s/MED_EK0_VSEmbedding.mat',save_path),'aps');

%     
%     
%     
%     % prepare the event training data
%     label = trnLabel(e, :);
%     trnDataX2 = trnDataX(:, label ~= 0);
%     label_train = label(label ~= 0);
%     
%     % prepare the event testing data
%     label_test = tstLabel(e, :);
%     tstDataX2 = tstDataX(:, label_test ~= 0);
%     label_test = label_test(label_test ~= 0);        
%     
%     % train a RBF kernel SVM event classifier
%     model = svmtrain(cast(label_train', 'double'), cast(trnDataX2', 'double'), '-t 2 -g 1 -q -c 100');    
%     
%     % apply the trained event classifier
%     [~, ~, conf] = svmpredict(zeros(size(tstDataX2(1, :)))', cast(tstDataX2', 'double'), model);    
%     
%     % calculate the average precision
%     aps(e) = calcMap(conf, label_test');
% end
% 
% %% Save Results
% if ~exist('./performance/','dir')
%     mkdir('./performance/');
% end
% 
% save('./performance/vs_vs_aps.mat','aps');