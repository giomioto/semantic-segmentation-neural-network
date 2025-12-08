import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# ================= CONFIGURAÇÕES DE DIRETÓRIOS =================
INPUT_SINGLE = Path('Imagens_um_Alimento')
INPUT_MULTI  = Path('Imagens_Varios_Alimentos')

OUTPUT_SINGLE = Path('Imagens_um_Alimento_Processadas')
OUTPUT_MULTI  = Path('Imagens_Varios_Alimentos_Processadas')
# ===============================================================


# =================== MÁSCARA BLINDADA ==========================
def criar_mascara_segmentacao_grabcut(img):
    """
    Pré-processamento BLINDADO:
    - Reduz tamanho para acelerar.
    - GrabCut com retângulo seguro (12% de margem).
    - Convex Hull para preencher falhas internas.
    - Corte circular para remover fundo/madeira/cantos.
    """
    h_orig, w_orig = img.shape[:2]

    # Redimensiona imagem para acelerar o GrabCut
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

    # GrabCut com margem reforçada
    margem_x = max(5, int(w * 0.12))
    margem_y = max(5, int(h * 0.12))
    rect = (margem_x, margem_y, w - 2*margem_x, h - 2*margem_y)

    mask = np.zeros(img_proc.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_proc, mask, rect, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_RECT)
    except:
        # fallback: caso o GrabCut falhe, considerar tudo como foreground
        return np.full((h_orig, w_orig), 255, dtype=np.uint8)

    mask_bin = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') * 255

    # Limpeza básica
    kernel = np.ones((3, 3), np.uint8)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=2)

    # Convex hull para fechar buracos
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        maior = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(maior)
        cv2.drawContours(mask_bin, [hull], -1, 255, -1)

    # Máscara circular (remove sobras da mesa)
    circular_mask = np.zeros_like(mask_bin)
    center = (int(w / 2), int(h / 2))
    radius = int(min(w, h) * 0.49)

    cv2.circle(circular_mask, center, radius, 255, -1)

    mask_final = cv2.bitwise_and(mask_bin, circular_mask)

    # Redimensionar de volta para tamanho original
    if scale != 1.0:
        mask_final = cv2.resize(mask_final, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    return mask_final


# =============== REALCE (opcional / desligado) =================
def aplicar_realce(img):
    """Aumenta contraste + nitidez. Mantém desligado por padrão."""
    if np.max(img) == 0:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_contraste = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # sharpening
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    return cv2.filter2D(img_contraste, -1, kernel)


# ===================== PIPELINE ================================
def processar_e_salvar(caminho_entrada, caminho_saida):
    try:
        img = cv2.imread(str(caminho_entrada))
        if img is None:
            return False

        mask = criar_mascara_segmentacao_grabcut(img)
        if np.max(mask) == 0:
            mask[:] = 255

        foreground = cv2.bitwise_and(img, img, mask=mask)

        # opicional:
        # foreground = aplicar_realce(foreground)

        cv2.imwrite(str(caminho_saida), foreground)
        return True

    except Exception as e:
        print("Erro:", e)
        return False


def task(args):
    entrada, saida = args
    saida.parent.mkdir(parents=True, exist_ok=True)
    return processar_e_salvar(entrada, saida)


def main():
    extensoes = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.JPG', '.PNG']
    tarefas = []

    # SINGLE-LABEL
    if INPUT_SINGLE.exists():
        print(f"Mapeando: {INPUT_SINGLE}")
        for ext in extensoes:
            for arq in INPUT_SINGLE.rglob(f'*{ext}'):
                rel = arq.relative_to(INPUT_SINGLE)
                dest = OUTPUT_SINGLE / rel
                tarefas.append((arq, dest))

    # MULTI-LABEL
    if INPUT_MULTI.exists():
        print(f"Mapeando: {INPUT_MULTI}")
        for arq in INPUT_MULTI.iterdir():
            if arq.suffix.lower() in extensoes:
                dest = OUTPUT_MULTI / arq.name
                tarefas.append((arq, dest))

    print(f"\nTotal de imagens: {len(tarefas)}")
    print("Iniciando pré-processamento BLINDADO...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(task, tarefas), total=len(tarefas)))

    print("\n✔ Finalizado!")
    print(f"Imagens salvas em:\n- {OUTPUT_SINGLE}\n- {OUTPUT_MULTI}")


if __name__ == "__main__":
    main()
