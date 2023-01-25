%%%--------------------------------------------------------------------
%%%  ANN merged with the firefly algorithm, for time series prediction
%%% 
%%% Citation:
%%% Mohammadi B. (2023). Modeling various drought time scales via a merged artificial neural
%%% network with a firefly algorithm, Hydrology, XX, XX-XX. 
%%% https://doi.org/10.3390/xxxxx
%%%
%%%--------------------------------------------------------------------
%%
clc;
clear;

% Load Data
Data = Load Data;
inputs = choose train phase; inputs = inputs';
targets = choose test sphase; targets = targets';

FeatureNum=size(inputs,1);
InputNum = size(inputs,2);
OutputNum = size(targets,2);

%% Create a Fitting Network
hiddenLayerSize = 1;    
TF={'tansig','purelin'};
netfa = newff(inputs,targets,hiddenLayerSize,TF);

netfa.inputs{1}.processFcns = {'removeconstantrows','mapminmax'}; 
netfa.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};
%rng ('default')
%net.divideFcn = 'dividerand';
%net.divideMode = 'sample'; 
%net.divideParam.trainRatio = 85/100;
%net.divideParam.valRatio = 15/100;
%net.divideParam.testRatio = 0/100;

trIndex =number of train;
tsIndex =number of test;
Xtr = inputs(:,trIndex);
Ytr = targets(:,trIndex); 

Xts = inputs(:,tsIndex);
Yts = targets(:,tsIndex);

[netfa] = TrainUsing_FA_Fcn(netfa,Xtr,Ytr);


outputs = netfa(inputs); 
errors = gsubtract(targets,outputs); 
performance = perform(netfa,targets,outputs);

trainInputs = inputs(:,trIndex);
trainOutputs= netfa(trainInputs);
trainTargets = targets(:,trIndex);
trainerrors=gsubtract(trainTargets,trainOutputs);    
trainperformance = perform(netfa,trainTargets,trainOutputs);  
trainMSE=mean(trainerrors.^2); 
trainRMSE=sqrt(trainMSE);
trainErrorMean=mean(trainerrors); 
trainErrorSTD=std(trainerrors); 

testInputs = inputs(:,tsIndex);
testOutputs=netfa(testInputs);
testTargets = targets(:,tsIndex);
testerrors=gsubtract(testTargets,testOutputs);
testperformance = perform(netfa,testTargets,testOutputs);
testMSE=mean(testerrors.^2);
testRMSE=sqrt(testMSE);
testErrorMean=mean(testerrors);
testErrorSTD=std(testerrors);


%% Plot Results

figure;
PlotResults(trainTargets,trainOutputs,'Train Data');   

figure;
PlotResults(testTargets,testOutputs,'Test Data');    

figure;
PlotResults(targets,outputs,'All Data');     


if ~isempty(which('plotregression'))
    figure;
    plotregression(trainTargets, trainOutputs, 'Train Data', ...
                   testTargets, testOutputs, 'Test Data', ...
                   targets, outputs, 'All Data');
    set(gcf,'Toolbar','figure');
end


view(netfa);

