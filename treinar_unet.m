% Treinamento U-Net
clear; clc; close all;

disp("Carregando datastores");
load("datastores.mat","dsTrain","dsVal");

% Configurações
classes     = ["fundo", "comida"];
numClasses  = numel(classes);
imageSize   = [256 256 3];

% Criar U-Net
lgraph = unetLayers(imageSize, numClasses);

% Treinamento
options = trainingOptions("adam", ...
    "ExecutionEnvironment", "auto", ...
    "InitialLearnRate",1e-3, ...
    "MaxEpochs",40, ...
    "MiniBatchSize",8, ...
    "Shuffle","every-epoch", ...
    "ValidationData",dsVal, ...
    "ValidationFrequency",50, ...
    "Verbose",true, ...
    "Plots","training-progress");

[net, info] = trainNetwork(dsTrain, lgraph, options);

save("modelo_unet.mat","net","info");

disp("Treinamento concluído com sucesso!");
