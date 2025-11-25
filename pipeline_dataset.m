% Gabriel Dutra Amaral, Giovanni Mioto
clc; clear; close all;

rootOriginal = "imagens";
rootFinal    = "dataset_final";

categorias = ["um_alimento", "varios_alimentos"];
imageSize  = [256 256];

% gerar as mascaras
disp("Gerando dataset_final");
for g = categorias
    disp("Processando grupo: " + g);

    outImgDir  = fullfile(rootFinal, g, "images");
    outMaskDir = fullfile(rootFinal, g, "masks");
    if ~exist(outImgDir, "dir"), mkdir(outImgDir); end
    if ~exist(outMaskDir, "dir"), mkdir(outMaskDir); end

    entradas = dir(fullfile(rootOriginal, g, "*"));
    entradas = entradas([entradas.isdir] & ~startsWith({entradas.name},"."));

    contador = 0;

    for folder = entradas'
        pastaAtual = fullfile(folder.folder, folder.name);
        imgs = dir(fullfile(pastaAtual, "*.jpg"));

        for i = 1:numel(imgs)
            contador = contador + 1;

            I  = imread(fullfile(imgs(i).folder, imgs(i).name));
            
            % Usar a nova função de segmentação baseada em PDI
            bw = generate_mask(I); % 0 = fundo, 255 = alimento

            newName = sprintf("%06d.png", contador);

            imwrite(I, fullfile(outImgDir, newName));
            imwrite(bw, fullfile(outMaskDir, newName));
        end
    end
end

disp("✔ Máscaras e imagens geradas!");


% cria os datastores
disp("🔧 Criando datastores...");

dsTrainAll = {};
dsValAll   = {};

for g = categorias
    imgDir  = fullfile(rootFinal, g, "images");
    maskDir = fullfile(rootFinal, g, "masks");

    % imagens
    imds = imageDatastore(imgDir);

    % mascaras (2 classes)
    classNames = ["fundo", "comida"];
    labelIDs   = [0 255];

    pxds = pixelLabelDatastore(maskDir, classNames, labelIDs);

    % split
    N = numel(imds.Files);
    idx = randperm(N);
    nTrain = round(0.8*N);

    imdsTrain = subset(imds, idx(1:nTrain));
    imdsVal   = subset(imds, idx(nTrain+1:end));

    pxdsTrain = subset(pxds, idx(1:nTrain));
    pxdsVal   = subset(pxds, idx(nTrain+1:end));

    % combinação
    dsTrain = combine(imdsTrain, pxdsTrain);
    dsVal   = combine(imdsVal, pxdsVal);

    % resize
    dsTrain = transform(dsTrain, @(d) resizeSeg(d, imageSize));
    dsVal   = transform(dsVal,   @(d) resizeSeg(d, imageSize));

    dsTrainAll{end+1} = dsTrain;
    dsValAll{end+1}   = dsVal;
end

% concatenação das duas categorias
dsTrain = combine(dsTrainAll{:});
dsVal   = combine(dsValAll{:});

save("datastores.mat","dsTrain","dsVal");

disp("✔ Todos datastores prontos!");

% função auxiliar para redimensionar uma imagem e sua máscara de segmentação

% A função generate_mask_pdi deve ser salva em um arquivo separado ou incluída aqui.
% Como já foi salva em generate_mask_pdi.m, vamos apenas garantir que ela seja chamada.
function out = resizeSeg(data, sz)
    I = imresize(data{1}, sz);
    M = imresize(data{2}, sz, "nearest");
    out = {I, M};
end
