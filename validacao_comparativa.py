import os
import glob
import pandas as pd
import google.generativeai as genai
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from collections import defaultdict
import time
from tqdm import tqdm

try:
    from preprocessing_augmentation import criar_mascara_segmentacao_grabcut, aplicar_realce
    print("Módulo de pré-processamento carregado com sucesso!")
except ImportError:
    print("ERRO CRÍTICO: Não foi possível importar preprocessing_augmentation.py. O arquivo deve estar no mesmo diretório.")
    # Funções mock para evitar erro, mas o usuário deve fornecer as reais
    def criar_mascara_segmentacao_grabcut(img_cv2):
        return np.full(img_cv2.shape[:2], 255, dtype=np.uint8)
    def aplicar_realce(img_cv2):
        return img_cv2

# --- CONFIGURAÇÃO DA API DO GOOGLE GEMINI ---
GOOGLE_API_KEY = "AIzaSyAz8EoHp53LC6rj92PWBbQcJ8TwIhcQxpk" 
genai.configure(api_key=GOOGLE_API_KEY)

# --- CONFIGURAÇÕES DO SEU MODELO  ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Alface', 'Almondega', 'Arroz', 'BatataFrita', 'Beterraba', 
               'BifeBovinoChapa', 'CarneBovinaPanela', 'Cenoura', 'FeijaoCarioca', 
               'Macarrao', 'Maionese', 'PeitoFrango', 'PureBatata', 
               'StrogonoffCarne', 'StrogonoffFrango', 'Tomate']
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = 'outputs_multilabel/best_single.pth'
IMG_DIR = 'Imagens_Varios_Alimentos'  # Pasta com as imagens de teste
THRESHOLD = 0.60
MIN_CONTOUR_AREA = 500

# --- CARREGAMENTO DO SEU MODELO ---
def load_my_model():
    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Erro: Modelo não encontrado em {MODEL_PATH}")
        return None
    model.to(DEVICE)
    model.eval()
    return model

my_model = load_my_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_my_model(pil_img):
    if my_model is None: return []
    
    # Conversão e Pré-processamento
    img_np = np.array(pil_img)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Segmentação
    mascara = criar_mascara_segmentacao_grabcut(img_cv2)
    if np.max(mascara) == 0: mascara[:] = 255
    
    # Encontrar contorno principal
    contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Grid 3x3
    detected_classes = set()
    col_step, row_step = w / 3, h / 3
    
    for i in range(3):
        for j in range(3):
            x1 = int(x + i * col_step)
            y1 = int(y + j * row_step)
            x2 = int(x + (i + 1) * col_step)
            y2 = int(y + (j + 1) * row_step)
            
            if (x2 - x1) * (y2 - y1) > MIN_CONTOUR_AREA:
                crop = pil_img.crop((x1, y1, x2, y2))
                input_tensor = transform(crop).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    outputs = my_model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                
                top_conf, top_idx = torch.max(probs, 0)
                if top_conf.item() >= THRESHOLD:
                    detected_classes.add(CLASS_NAMES[top_idx])
                    
    return list(detected_classes)

# --- LÓGICA DE INFERÊNCIA DO GEMINI ---
def predict_gemini(pil_img):
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Prompt Restritivo: Força o Gemini a usar apenas as classes
    prompt = f"""
    Atue como um especialista em identificação de alimentos.
    Analise esta imagem e identifique quais alimentos estão presentes.
    
    IMPORTANTE: Você DEVE mapear os alimentos encontrados APENAS para a seguinte lista de classes exata:
    {CLASS_NAMES}
    
    Se houver um alimento na imagem que não se encaixa perfeitamente em nenhuma dessas classes (ex: brócolis, suco), ignore-o.
    Se houver "Feijão Preto" mas a lista só tem "FeijaoCarioca", use "FeijaoCarioca" se for visualmente similar, senão ignore.
    
    Retorne APENAS os nomes das classes encontrados, separados por vírgula, sem texto adicional.
    Exemplo de saída: Arroz, FeijaoCarioca, BifeBovinoChapa
    """
    
    try:
        response = model.generate_content([prompt, pil_img])
        text = response.text.strip()
        # Limpeza básica
        items = [x.strip() for x in text.split(',')]
        # Filtra para garantir que só retornou classes válidas
        valid_items = [x for x in items if x in CLASS_NAMES]
        return valid_items
    except Exception as e:
        print(f"Erro no Gemini: {e}")
        return []

# --- LOOP DE AVALIAÇÃO ---
def run_evaluation():
    image_paths = glob.glob(os.path.join(IMG_DIR, "*.*"))
    # Filtra extensões comuns
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    results_data = []
    
    print(f"Iniciando avaliação de {len(image_paths)} imagens...")
    
    for img_path in tqdm(image_paths):
        try:
            filename = os.path.basename(img_path)
            pil_img = Image.open(img_path).convert('RGB')
            
            # 1. Predição do Seu Modelo
            my_preds = set(predict_my_model(pil_img))
            
            # 2. Predição do Gemini (Ground Truth)
            # Pausa curta para não estourar cota gratuita do Gemini
            time.sleep(1.5) 
            gemini_preds = set(predict_gemini(pil_img))
            
            if not gemini_preds:
                # Se o Gemini não achou nada da lista, pulamos ou marcamos como vazio
                # Isso evita divisão por zero na acurácia se a imagem for ruim
                pass

            # 3. Comparação
            # Acertos (Intersecção): O que os dois concordam
            common = my_preds.intersection(gemini_preds)
            
            # Alucinações (Seu modelo viu, mas não estava lá segundo Gemini)
            false_positives = my_preds - gemini_preds
            
            # Omissões (Estava lá, mas seu modelo não viu)
            missed = gemini_preds - my_preds
            
            # Cálculo de Acurácia para esta imagem (Baseada no Gemini como verdade)
            # Fórmula: Acertos / Total de coisas que existem (segundo Gemini)
            # Se Gemini diz 0 coisas, e nós achamos 0, acurácia é 100%.
            if len(gemini_preds) > 0:
                accuracy_img = (len(common) / len(gemini_preds)) * 100
            else:
                # Se o Gemini não viu nada da lista, e seu modelo tbm não: 100%. Se seu modelo viu: 0%
                accuracy_img = 100.0 if len(my_preds) == 0 else 0.0

            results_data.append({
                "Arquivo": filename,
                "Gemini (Real)": ", ".join(gemini_preds),
                "Seu Modelo (Pred)": ", ".join(my_preds),
                "Em Comum (Acerto)": ", ".join(common),
                "Modelo Errou (Alucinação)": ", ".join(false_positives),
                "Modelo Não Viu (Omissão)": ", ".join(missed),
                "Acurácia (%)": round(accuracy_img, 2)
            })
            
        except Exception as e:
            print(f"Erro ao processar {img_path}: {e}")

    # --- RELATÓRIO FINAL ---
    df = pd.DataFrame(results_data)
    
    # Métricas Globais
    media_acuracia = df["Acurácia (%)"].mean()
    total_itens_gemini = df["Gemini (Real)"].apply(lambda x: len(x.split(',')) if x else 0).sum()
    total_acertos = df["Em Comum (Acerto)"].apply(lambda x: len(x.split(',')) if x else 0).sum()
    
    print("\n" + "="*40)
    print("RESULTADO DA AVALIAÇÃO COMPARATIVA")
    print("="*40)
    print(f"Total Imagens: {len(df)}")
    print(f"Acurácia Média por Imagem: {media_acuracia:.2f}%")
    # Uma métrica de Recall Global (Total de Acertos / Total de Itens Reais)
    recall_global = (total_acertos / total_itens_gemini * 100) if total_itens_gemini > 0 else 0
    print(f"Recall Global do Dataset: {recall_global:.2f}%")
    
    # Salvar
    output_file = "relatorio_comparativo_gemini.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\nRelatório detalhado salvo em: {output_file}")
    
    return df

if __name__ == "__main__":
    # Verifica se a pasta existe
    if not os.path.exists(IMG_DIR):
        print(f"Crie a pasta '{IMG_DIR}' e coloque imagens nela para testar.")
    else:
        df = run_evaluation()
        print(df.head())