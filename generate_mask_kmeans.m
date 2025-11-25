function mask = generate_mask_kmeans(I)
% GENERATE_MASK_KMEANS Gera uma máscara binária para a comida usando K-means.
% A função assume que a imagem contém 3 classes principais: fundo (mesa), prato e comida.

% 1. Converter a imagem para o espaço de cores L*a*b* para melhor separação de cores
lab_I = rgb2lab(I);

% 2. Usar apenas os canais 'a' e 'b' para a segmentação por cor
ab = lab_I(:,:,2:3);
ab = im2single(ab);

% 3. Segmentar a imagem em 3 clusters (fundo, prato, comida) usando K-means
nColors = 3;
pixel_labels = imsegkmeans(ab, nColors, 'NumAttempts', 3);

% 4. Analisar os clusters para identificar a comida.
% O cluster da comida geralmente será o menor em área e mais centralizado.
% Para simplificar, vamos assumir que o cluster da comida é o que tem a menor área
% E que não é o fundo (que é o maior cluster).

stats = regionprops(pixel_labels, 'Area', 'PixelIdxList');
areas = [stats.Area];

% Encontrar o cluster com a maior área (provavelmente o fundo)
[~, idx_fundo] = max(areas);

% Inicializar a máscara como falsa (tudo é fundo)
mask = false(size(pixel_labels));

% Iterar sobre os clusters, ignorando o maior (fundo)
for k = 1:nColors
    if k ~= idx_fundo
        % Criar uma máscara temporária para o cluster atual
        temp_mask = (pixel_labels == k);
        
        % O prato é geralmente um anel ou círculo grande. A comida é uma forma menor.
        % Vamos usar a área para tentar distinguir o prato da comida.
        % O cluster restante com a menor área é provavelmente a comida.
        
        % Encontrar o cluster que não é o fundo e tem a menor área.
        % Esta é uma heurística, mas funciona bem para este tipo de imagem.
        
        % Se houver apenas 2 clusters restantes (prato e comida),
        % o menor é a comida. Se houver apenas 1 (prato OU comida), é esse.
        
        % Uma heurística mais robusta seria:
        % 1. Identificar o fundo (maior área).
        % 2. Dos restantes, o prato é o que tem o formato mais circular e maior área.
        % 3. O que sobrar é a comida.
        
        % Simplificando: vamos tentar o cluster com a menor área entre os não-fundo.
        
        % Para evitar a complexidade de regionprops em cada iteração,
        % vamos identificar os índices dos clusters não-fundo.
        
        non_background_indices = setdiff(1:nColors, idx_fundo);
        
        % Se houver 2 clusters não-fundo (prato e comida)
        if numel(non_background_indices) == 2
            % Comparar as áreas dos dois clusters não-fundo
            area1 = stats(non_background_indices(1)).Area;
            area2 = stats(non_background_indices(2)).Area;
            
            if area1 < area2
                idx_comida = non_background_indices(1);
            else
                idx_comida = non_background_indices(2);
            end
            
            mask = (pixel_labels == idx_comida);
            break; % Encontramos a comida, podemos sair do loop
            
        % Se houver apenas 1 cluster não-fundo (apenas comida ou apenas prato)
        elseif numel(non_background_indices) == 1
            % Assumimos que é a comida (ou o que queremos segmentar)
            mask = (pixel_labels == non_background_indices(1));
            break;
        end
    end
end

% Se a heurística falhar (o que é comum), vamos tentar uma abordagem mais simples:
% O cluster da comida é o que tem a menor área total.
if ~any(mask(:))
    [~, idx_comida] = min(areas);
    % Se o menor cluster for o fundo, pegamos o segundo menor
    if idx_comida == idx_fundo
        areas(idx_fundo) = inf; % Ignora o fundo
        [~, idx_comida] = min(areas);
    end
    mask = (pixel_labels == idx_comida);
end

% A máscara final deve ser binária (0 ou 1)
mask = uint8(mask) * 255;

end
