%% Script to Load VideoStory Data

%% Load VideoSotry Data
load('/import/vision-datasets3/VideoStory/vs/datasets/VideoStory46K/feature_mbh.mat');

%% Load Term Vector
load('/import/vision-datasets3/VideoStory/vs/datasets/VideoStory46K/tv.mat');

X = data';
Y = tv';
clear data tv;