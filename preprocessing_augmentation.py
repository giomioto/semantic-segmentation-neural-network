import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# ================= CONFIGURAÇÕES DE DIRETÓRIOS =================
INPUT_SINGLE = Path('Imagens_um_Alimento')
INPUT_MULTI = Path('Imagens_Varios_Alimentos')
OUTPUT_SINGLE = Path('Imagens_um_Alimento_Processadas2')
OUTPUT_MULTI = Path('Imagens_Varios_Alimentos_Processadas2')
# ===============================================================

def criar_mascara_segmentacao_grabcut(img):
    """
    Versão BLINDADA:
    1. Margem maior (ensina pro GrabCut que a borda é fundo).
    2. GrabCut + Convex Hull (tapa buracos).
    3. Guilhotina Circular (corta sobras de mesa nos cantos).
    """
    h_orig, w_orig = img.shape[:2]
    
    # --- 1. Redimensionar (Otimização) ---
    MAX_DIM = 512
    scale = 1.0
    
    if max(h_orig, w_orig) > MAX_DIM:
        scale = MAX_DIM / max(h_orig, w_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        img_proc = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_proc = img.copy()

    h, w = img_proc.shape[:2]

    # --- 2. GrabCut com MARGEM MAIOR ---
    # Aumentei de 0.05 para 0.12 (12% de margem).
    # Isso obriga o algoritmo a pegar amostras da madeira/toalha e classificar como FUNDO.
    margem_x = max(1, int(w * 0.12))
    margem_y = max(1, int(h * 0.12))
    rect = (margem_x, margem_y, w - 2*margem_x, h - 2*margem_y)

    mask = np.zeros(img_proc.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_proc, mask, rect, bgdModel, fgdModel, iterCount=4, mode=cv2.GC_INIT_WITH_RECT)
    except:
        return np.full((h_orig, w_orig), 255, dtype=np.uint8)

    mask_binaria = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask_final_small = mask_binaria * 255

    # Limpeza básica
    kernel = np.ones((3, 3), np.uint8) 
    mask_final_small = cv2.morphologyEx(mask_final_small, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- 3. CONVEX HULL (Tapar buraco do meio) ---
    contours, _ = cv2.findContours(mask_final_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        maior_contorno = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(maior_contorno)
        cv2.drawContours(mask_final_small, [hull], -1, 255, -1)

    # --- 4. GUILHOTINA CIRCULAR (O CORTE FINAL) ---
    # Como as imagens são de pratos, vamos criar um círculo no centro da imagem.
    # Tudo que estiver fora desse círculo será apagado. Isso mata os cantos da mesa.
    
    # Cria uma máscara preta
    circular_mask = np.zeros_like(mask_final_small)
    
    # Define o centro e um raio (95% da menor dimensão para não cortar o prato se ele for grande)
    center = (int(w / 2), int(h / 2))
    radius = int(min(h, w) / 2 * 0.98) 
    
    # Desenha um círculo branco
    cv2.circle(circular_mask, center, radius, 255, -1)
    
    # Aplica a guilhotina: Só mantem o que está dentro do círculo E foi detectado pelo GrabCut
    mask_final_small = cv2.bitwise_and(mask_final_small, circular_mask)

    # --- Redimensionar de volta ---
    if scale != 1.0:
        mask_final = cv2.resize(mask_final_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    else:
        mask_final = mask_final_small

    return mask_final

def aplicar_realce(img):
    if np.max(img) == 0: return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_contraste = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    kernel_sharpening = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
    img_sharp = cv2.filter2D(img_contraste, -1, kernel_sharpening)
    return img_sharp

def processar_e_salvar(caminho_entrada, caminho_saida):
    try:
        img = cv2.imread(str(caminho_entrada))
        if img is None: return False

        mascara = criar_mascara_segmentacao_grabcut(img)
        if np.max(mascara) == 0: mascara[:] = 255

        fundo_removido = cv2.bitwise_and(img, img, mask=mascara)
        img_realcada = aplicar_realce(fundo_removido)
        img_final = cv2.bitwise_and(img_realcada, img_realcada, mask=mascara)

        cv2.imwrite(str(caminho_saida), img_final)
        return True
    except Exception as e:
        print(f"Erro: {e}")
        return False

def wrapper_processamento(args):
    entrada, saida = args
    saida.parent.mkdir(parents=True, exist_ok=True)
    return processar_e_salvar(entrada, saida)

def main():
    extensoes = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.JPG', '.PNG']
    tarefas = []

    if INPUT_SINGLE.exists():
        print(f"Mapeando: {INPUT_SINGLE}...")
        for ext in extensoes:
            for arq in INPUT_SINGLE.rglob(f'*{ext}'):
                rel = arq.relative_to(INPUT_SINGLE)
                dest = OUTPUT_SINGLE / rel
                tarefas.append((arq, dest))
    
    if INPUT_MULTI.exists():
        print(f"Mapeando: {INPUT_MULTI}...")
        for arq in INPUT_MULTI.iterdir():
            if arq.suffix.lower() in extensoes:
                dest = OUTPUT_MULTI / arq.name
                tarefas.append((arq, dest))

    total = len(tarefas)
    WORKERS = 4 
    
    print(f"\nIniciando processamento BLINDADO em {total} imagens.")
    print(f"Estratégia: Margem Alta (12%) + Guilhotina Circular.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
        list(tqdm(executor.map(wrapper_processamento, tarefas), total=total))

    print("\nProcessamento Concluído!")

if __name__ == "__main__":
    main()