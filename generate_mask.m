function mask = generate_mask_pdi(I)
% GENERATE_MASK_PDI Gera uma máscara binária para a comida usando PDI.
% A estratégia é isolar o prato (objeto grande e central) e, em seguida,
% isolar a comida (objeto de cor diferente do prato).

% 1. Pré-processamento: Converter para escala de cinza e aplicar filtro para realçar bordas
gI = rgb2gray(I);
gI = imgaussfilt(gI, 2); % Suaviza a imagem

% 2. Segmentação do Prato (Objeto Central e Claro)
% O prato é geralmente mais claro que o fundo (mesa).
% Usamos uma limiarização simples para isolar objetos claros.
T = graythresh(gI);
bw_prato_candidato = imbinarize(gI, T);

% 3. Limpeza e Análise de Componentes Conectados (para isolar o prato)
% Remove pequenos ruídos
bw_prato_candidato = bwareaopen(bw_prato_candidato, 500); 

% Preenche buracos (a comida dentro do prato)
bw_prato_candidato = imfill(bw_prato_candidato, 'holes');

% Encontra o maior componente conectado (que deve ser o prato)
stats = regionprops(bw_prato_candidato, 'Area', 'PixelIdxList');
areas = [stats.Area];

if isempty(areas)
    mask = zeros(size(gI), 'uint8');
    return;
end

[~, idx_prato] = max(areas);
mask_prato = false(size(gI));
mask_prato(stats(idx_prato).PixelIdxList) = true;

% 4. Segmentação da Comida (Dentro da Máscara do Prato)
% A comida é o que tem cor diferente do prato (branco/claro) dentro da área do prato.
% A nova lógica é: Comida = (Tudo dentro do prato) MENOS (A área do prato que é branca/muito clara)

% Isolamos a área branca/clara (prato) dentro da máscara do prato.
% Usamos o canal L (luminosidade) do L*a*b* para isolar o branco.
lab_I = rgb2lab(I);
L_channel = lab_I(:,:,1);

% Um limiar alto no canal L (ex: L > 85) isola o branco do prato.
% O valor 85 é uma heurística para pratos brancos.
bw_prato_branco = (L_channel > 75) & mask_prato; % Diminuindo o limiar para capturar mais do prato branco

% Limpeza: Remove a comida branca (arroz, purê) que pode ter sido incluída
% na máscara do prato branco, pois a comida branca é geralmente menor.
bw_prato_branco = bwareaopen(bw_prato_branco, 2000); % Aumentando o filtro de área para garantir que apenas o prato grande permaneça

% Subtração: A máscara da comida é a máscara do prato menos a área do prato branco.
% Usamos a operação XOR (ou subtração lógica)
mask_comida = mask_prato & ~bw_prato_branco;

% 5. Limpeza Final
% Remove pequenos ruídos e preenche buracos na comida
mask_comida = bwareaopen(mask_comida, 50); % Remove pequenos ruídos que sobraram
mask_comida = imfill(mask_comida, 'holes'); % Preenche buracos na comida

% A máscara final deve ser binária (0 ou 1)
mask = uint8(mask_comida) * 255;

end
