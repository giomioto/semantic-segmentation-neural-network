import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'outputs_multilabel/best_single.pth'
TEST_DIR = Path('Imagens_Varios_Alimentos_Processadas')

CLASS_NAMES = ['Alface', 'Almondega', 'Arroz', 'BatataFrita', 'Beterraba', 
               'BifeBovinoChapa', 'CarneBovinaPanela', 'Cenoura', 'FeijaoCarioca', 
               'Macarrao', 'Maionese', 'PeitoFrango', 'PureBatata', 
               'StrogonoffCarne', 'StrogonoffFrango', 'Tomate']

CONFIDENCE_THRESHOLD = 0.60
MIN_CONTOUR_AREA = 500

def load_model():
    print(f"Carregando modelo...")
    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, len(CLASS_NAMES))
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"ERRO: Modelo não encontrado em {MODEL_PATH}")
        exit()
        
    model.to(DEVICE)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def criar_mascara_segmentacao_grabcut(img):
    h, w = img.shape[:2]
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (5, 5, w-10, h-10)
    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_RECT)
        return np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') * 255
    except:
        return np.full((h, w), 255, dtype=np.uint8)

def predict_multilabel_strategy(model, image_path):
    img_cv2 = cv2.imread(str(image_path))
    if img_cv2 is None: return []
    
    mask = criar_mascara_segmentacao_grabcut(img_cv2)
    img_masked = cv2.bitwise_and(img_cv2, img_cv2, mask=mask)
    img_pil = Image.fromarray(cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)

    crops = []
    col_step, row_step = w / 3, h / 3
    for i in range(3):
        for j in range(3):
            x1 = int(x + i * col_step)
            y1 = int(y + j * row_step)
            x2 = int(x + (i + 1) * col_step)
            y2 = int(y + (j + 1) * row_step)
            if (x2 - x1) * (y2 - y1) > MIN_CONTOUR_AREA:
                crops.append(img_pil.crop((x1, y1, x2, y2)))

    if not crops: return []
    batch = torch.stack([transform(c) for c in crops]).to(DEVICE)
    
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
    
    detected = set()
    max_probs, max_indices = torch.max(probs, dim=1)
    for conf, idx in zip(max_probs, max_indices):
        if conf.item() >= CONFIDENCE_THRESHOLD:
            detected.add(CLASS_NAMES[idx.item()])
            
    return list(detected)

def main():
    print("=== VALIDAÇÃO MANUAL (DICIONÁRIO) ===")
    
    GABARITO = {
        "IMG_20140922_141541933.jpg": ["Arroz", "Maionese", "StrogonoffCarne"],
        "IMG_20140922_141953922.jpg": ["Arroz", "Maionese", "PureBatata", "Alface", "PeitoFrango"],
        "IMG_20140923_141839167.jpg": ["BatataFrita", "Alface", "Tomate", "StrogonoffCarne"],
        "IMG_20140924_140220363.jpg": ["Cenoura", "Beterraba"],
        "IMG_20140924_140725750.jpg": ["Cenoura", "Beterraba", "Alface"],
        "IMG_20140926_140501433.jpg": ["Beterraba", "BatataFrita", "Arroz", "Tomate", "StrogonoffFrango", "PureBatata"],
        "IMG_20141114_135700043.jpg": ["Almondega", "StrogonoffCarne", "FeijaoCarioca", "BatataFrita"],
        "IMG_20141120_122214184.jpg": ["BatataFrita", "StrogonoffCarne", "Arroz", "PureBatata"],
        "IMG_20150305_142018869.jpg": ["Tomate"],
        "IMG_20150323_140956006.jpg": ["StrogonoffFrango", "PureBatata"],
        "prato_4_4.jpg": ["PeitoFrango", "Cenoura", "Almondega"]
    }

    model = load_model()
    y_true = []
    y_pred = []
    dados_para_csv = []
    
    print(f"Iniciando avaliação de {len(GABARITO)} imagens...")

    for nome_arquivo, rotulos_reais in tqdm(GABARITO.items()):
        caminho = TEST_DIR / nome_arquivo
        
        if not caminho.exists():
            print(f"\n[AVISO] Imagem não encontrada: {nome_arquivo} (Verifique o nome ou a pasta)")
            continue
            
        preditos = predict_multilabel_strategy(model, caminho)
        
        y_true.append(rotulos_reais)
        y_pred.append(preditos)

        dados_para_csv.append({
            "Arquivo": nome_arquivo,
            "Real": ", ".join(rotulos_reais),
            "Predito_IA": ", ".join(preditos),
            "Acertou_Tudo": "SIM" if set(rotulos_reais) == set(preditos) else "NAO"
        })
        
        print(f"\nArquivo: {nome_arquivo}")
        print(f"  - Real: {rotulos_reais}")
        print(f"  - IA Viu: {preditos}")

    if len(y_true) > 0:
        mlb = MultiLabelBinarizer(classes=CLASS_NAMES)
        y_true_bin = mlb.fit_transform(y_true)
        y_pred_bin = mlb.transform(y_pred)
        
        print("\n" + "="*60)
        print("RESULTADO FINAL")
        print("="*60)
        print(classification_report(y_true_bin, y_pred_bin, target_names=CLASS_NAMES, zero_division=0))
        
        exact_match = np.all(y_true_bin == y_pred_bin, axis=1).mean()
        print(f"Acurácia Exata (Acertou o prato 100%): {exact_match:.2%}")
    else:
        print("\nNenhuma imagem válida foi processada. Verifique os caminhos.")

    print("\n=== SALVANDO ARQUIVOS ===")
    try:
        df = pd.DataFrame(dados_para_csv)
        df.to_csv("validacao_detalhada.csv", index=False, encoding='utf-8-sig')
        print("Tabela salva: 'validacao_detalhada.csv'")
    except Exception as e:
        print(f"Erro ao salvar CSV: {e}")

    try:
        report_str = classification_report(y_true_bin, y_pred_bin, target_names=CLASS_NAMES, zero_division=0)
        
        with open("relatorio_metricas.txt", "w", encoding='utf-8') as f:
            f.write("RELATÓRIO DE AVALIAÇÃO - TASK MULTILABEL (GRADE 3x3)\n")
            f.write("==================================================\n\n")
            f.write(report_str)
            f.write(f"\n\nAcurácia Exata (Prato Completo): {exact_match:.2%}\n")
        print("Relatório salvo: 'relatorio_metricas.txt'")
    except Exception as e:
        print(f"Erro ao salvar TXT: {e}")

if __name__ == "__main__":
    main()