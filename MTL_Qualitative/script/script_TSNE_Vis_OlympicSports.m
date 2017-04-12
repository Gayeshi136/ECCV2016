%% script to visualize test data embedding for Olympic Sports
clear -except select_class;
close all;

addpath('/export/experiments/Alex/IJCV16/code/SharedTools/function/');
addpath('/export/experiments/Alex/Toolbox/Matlab/tSNE_matlab/');
addpath('/export/experiments/Alex/Toolbox/Matlab/ExportFigure/');

%% Load Class Names
EmbeddingMethod = 'add';
labelvector_path = '/export/experiments-data/Alex/IJCV16/OlympicSports/';
temp = load(sprintf([labelvector_path,'ClassLabelPhraseDict_mth-%s.mat'],EmbeddingMethod));
ClassLabels = temp.ClassLabelsPhrase;

if ~exist('select_class','var')
    select_class = sort(randsample(1:8,5,false),'ascend');
end

SelectClassLabels = ClassLabels(select_class);

%% Load Ridge Regression data
temp = load('/export/experiments-data/Alex/ECCV16/MTL_Qualitative/OlympicSportsTest_mdl-RR_trial-1.mat');
RR_Z_ts = temp.S_ts;
RR_ClassNo = temp.ts_ClassNo;
RR_Prototype = temp.Prototype;
RR_Prototype = RR_Prototype(select_class,:);
UniqueClass = unique(RR_ClassNo);
UniqueClass = UniqueClass(select_class);
data_ind = ismember(RR_ClassNo,UniqueClass);
RR_ClassNo = RR_ClassNo(data_ind);
RR_Z_ts = RR_Z_ts(data_ind,:);

%% Load VideoStory Model data
temp = load('/export/experiments-data/Alex/ECCV16/MTL_Qualitative/OlympicSportsTest_mdl-VS_trial-1.mat');
VS_Z_ts = (temp.S_ts);
VS_ClassNo = temp.ts_ClassNo;
VS_Prototype = temp.Prototype;
VS_Prototype = VS_Prototype(select_class,:);
data_ind = ismember(VS_ClassNo,UniqueClass);
VS_ClassNo = VS_ClassNo(data_ind);
VS_Z_ts = VS_Z_ts(data_ind,:);


numProto = size(RR_Prototype,1);
TargetClassLabels = ClassLabels(UniqueClass);
%% Copmute Affinity Matrix
RR_Aug_Z = func_L2Normalization([RR_Prototype ; RR_Z_ts]);
K_RR = RR_Aug_Z*RR_Aug_Z';

VS_Aug_Z = func_L2Normalization([VS_Prototype ; VS_Z_ts]);
K_VS = VS_Aug_Z*VS_Aug_Z';

%% Do TSNE Projection
X_RR = tsne(RR_Aug_Z,[],2,[],20);
X_VS = tsne(VS_Aug_Z,[],2,[],20);

Proto_RR = X_RR(1:numProto,:);
X_RR = X_RR(numProto+1:end,:);

Proto_VS = X_VS(1:numProto,:);
X_VS = X_VS(numProto+1:end,:);

%% Plot Data Points
backgroundcolor = [210 210 210]/255;
prototype_text_fontsize = 13;

figure('position',[200 200 700 360]);
color = colormap(hsv(numProto));

for i = 1:numel(UniqueClass)
    text(Proto_RR(i,1)+2,Proto_RR(i,2)-7, SelectClassLabels{i}, ...
        'HorizontalAlignment', 'Center', ...
        'FontUnits', 'pixels', ...
        'FontSize', prototype_text_fontsize,'color','black','String',TargetClassLabels{i},'BackgroundColor',backgroundcolor);hold on;
end


for i = 1:numel(UniqueClass)
    
    idx = RR_ClassNo == UniqueClass(i);
    plot(X_RR(idx,1),X_RR(idx,2),'.','color',color(i,:),'markersize',10); hold on;

    plot(Proto_RR(i,1),Proto_RR(i,2),'p','color',color(i,:),...
        'markersize',15,'markerfacecolor',color(i,:),'markeredgecolor','black','linewidth',2);

end

xlim = [1.1*min(X_RR(:,1)) 1.1*max(X_RR(:,1))];
set(gca,'xlim',xlim);
ylim = [1.1*min(X_RR(:,2)) 1.1*max(X_RR(:,2))];
set(gca,'ylim',ylim);
axis off;


fig_filepath = '/export/experiments-data/Alex/ECCV16/MTL_Qualitative/Figure/OlympicSports_TSNE_RR.pdf';
export_fig(fig_filepath,'-transparent');


%% Multi-Task Embedding
figure('position',[200 200 700 360]);
color = colormap(hsv(numProto));

for i = 1:numel(UniqueClass)
    text(Proto_RR(i,1)+2,Proto_RR(i,2)-7, SelectClassLabels{i}, ...
        'HorizontalAlignment', 'Center', ...
        'FontUnits', 'pixels', ...
        'FontSize', prototype_text_fontsize,'color','black','String',TargetClassLabels{i},'BackgroundColor',backgroundcolor);hold on;
end


for i = 1:numel(UniqueClass)
    
    idx = RR_ClassNo == UniqueClass(i);
    plot(X_VS(idx,1),X_VS(idx,2),'.','color',color(i,:),'markersize',10); hold on;
    
    plot(Proto_VS(i,1),Proto_VS(i,2),'p','color',color(i,:),...
        'markersize',15,'markerfacecolor',color(i,:),'markeredgecolor','black','linewidth',2);
    
end

xlim = [1.1*min(X_VS(:,1)) 1.1*max(X_VS(:,1))];
set(gca,'xlim',xlim);
ylim = [1.1*min(X_VS(:,2)) 1.1*max(X_VS(:,2))];
set(gca,'ylim',ylim);
axis off;

fig_filepath = '/export/experiments-data/Alex/ECCV16/MTL_Qualitative/Figure/OlympicSports_TSNE_VS.pdf';
export_fig(fig_filepath,'-transparent');