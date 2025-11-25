% Testar U-Net treinada em uma imagem nova
clear; clc; close all;

% Carregar modelo treinado
load("modelo_unet.mat","net");

% Caminho da imagem a testar
[arq, caminho] = uigetfile({'*.jpg;*.png'}, 'Selecione uma imagem para testar');
if isequal(arq,0)
    disp("Nenhuma imagem selecionada."); 
    return;
end

imgPath = fullfile(caminho, arq);
I = imread(imgPath);

figure; imshow(I); title("Imagem original");

% Preparar imagem para a U-Net
inputSize = net.Layers(1).InputSize(1:2);
Iresized = imresize(I, inputSize);

% Fazer a segmentação
pred = semanticseg(Iresized, net);

% Converter para máscara binária
mask = pred == "comida";

figure; imshow(mask); title("Máscara gerada pela U-Net");

% Criar overlay (máscara sobreposta à imagem)
cmap = [
    1 0 0;   % classe 1: comida (vermelho)
    0 1 0    % classe 2: fundo   (verde)
];

overlay = labeloverlay(Iresized, pred, ...
    "Transparency", 0.4, ...
    "Colormap", cmap);

figure; imshow(overlay);
title("Segmentação sobreposta");

% recortar apenas a comida
recorte = Iresized;
recorte(repmat(~mask,[1 1 3])) = 0;

figure; imshow(recorte);
title("Objeto segmentado (comida isolada)");

%% CLASSIFICAÇÃO DO ALIMENTO SEGMENTADO
% Esta seção requer um modelo de classificação treinado (ex: AlexNet, ResNet)
% que deve ser carregado aqui.

% Exemplo de como carregar um modelo de classificação (substitua pelo seu)
% load("modelo_classificacao.mat", "netClassificacao");

% 1. Preparar o recorte para a classificação
% O recorte (recorte) já contém apenas a comida.
% É necessário redimensioná-lo para o tamanho de entrada da rede de classificação.
% Exemplo:
% inputSizeClass = netClassificacao.Layers(1).InputSize(1:2);
% recorteClass = imresize(recorte, inputSizeClass);

% 2. Classificar o recorte
% [label, scores] = classify(netClassificacao, recorteClass);

% 3. Exibir o resultado
% disp(['Alimento Classificado: ', char(label)]);
% disp(['Confiança: ', num2str(max(scores)*100), '%']);

% Como não temos o modelo de classificação, vamos apenas exibir uma mensagem
disp(" ");
disp("--- CLASSIFICAÇÃO ---");
disp("A classificação do tipo de alimento requer um modelo de classificação treinado.");
disp("Use o recorte da comida isolada (variável 'recorte') como entrada para seu modelo de classificação.");

%% CÁLCULO DAS MÉTRICAS DE AVALIAÇÃO (IoU e Dice Coefficient)

disp(" ");
disp("--- CÁLCULO DAS MÉTRICAS ---");

% 1. Selecionar a Máscara de Verdade (Ground Truth)
[arq_gt, caminho_gt] = uigetfile({'*.png'}, 'Selecione a Mascara de Verdade (Ground Truth) correspondente');
if isequal(arq_gt,0)
    disp("Máscara de Verdade não selecionada. Não é possível calcular as métricas."); 
else
    gtMaskPath = fullfile(caminho_gt, arq_gt);
    gtMask = imread(gtMaskPath);
    
    % A máscara de verdade precisa ser redimensionada para o mesmo tamanho da previsão
    gtMask = imresize(gtMask, inputSize, "nearest");
    
    % Para simplificar, vamos binarizar a máscara de verdade para 0 e 1
    % Assumindo que a máscara de verdade tem 0 para fundo e 255 para comida
    gtMaskBin = logical(gtMask > 0);
    
    % A previsão 'pred' é uma matriz categorical. Vamos convertê-la para binária (0 e 1)
    predBin = logical(pred == "comida");
    
    % 2. Calcular o IoU (Intersection over Union) e Dice Coefficient
    % O MATLAB tem as funções jaccard e dice para isso.
    
    % IoU (Jaccard Index)
    iou = jaccard(predBin, gtMaskBin);
    
    % Dice Coefficient (F-score)
    dice_coeff = dice(predBin, gtMaskBin);
    
    disp(['IoU (Intersection over Union) para a classe "comida": ', num2str(iou)]);
    disp(['Dice Coefficient (F-score) para a classe "comida": ', num2str(dice_coeff)]);
    
    % Exibir a máscara de verdade para comparação
    figure; imshow(gtMaskBin); title("Máscara de Verdade (Ground Truth)");
end
