import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import cv2 

# --- IMPORTANDO SEU SCRIPT DE PRÉ-PROCESSAMENTO ---
try:
    from preprocessing_augmentation import criar_mascara_segmentacao_grabcut, aplicar_realce
    print("Módulo de pré-processamento carregado com sucesso!")
except ImportError:
    print("ERRO: Não foi possível importar preprocessing_augmentation.py.")

# --- 1. CONFIGURAÇÕES ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Alface', 'Almondega', 'Arroz', 'BatataFrita', 'Beterraba', 
               'BifeBovinoChapa', 'CarneBovinaPanela', 'Cenoura', 'FeijaoCarioca', 
               'Macarrao', 'Maionese', 'PeitoFrango', 'PureBatata', 
               'StrogonoffCarne', 'StrogonoffFrango', 'Tomate']
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = 'outputs_multilabel/best_ml.pth'

# --- 2. CARREGAR MODELO ---
def get_model():
    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, NUM_CLASSES)
    return model

print("Carregando modelo...")
model = get_model()
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Modelo carregado com sucesso!")
except FileNotFoundError:
    print(f"ERRO CRÍTICO: Não achei o arquivo {MODEL_PATH}.")

model.to(DEVICE)
model.eval()

# --- 3. TRANSFORMAÇÃO FINAL (Para a IA) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. FUNÇÃO DE PRÉ-PROCESSAMENTO (A Ponte) ---
def executar_pipeline_blindado(imagem_pil):
    """
    Converte PIL -> OpenCV, aplica seu script, converte OpenCV -> PIL
    """
    # 1. Converter PIL (RGB) para Numpy/OpenCV (BGR)
    img_np = np.array(imagem_pil)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 2. Aplicar a lógica do seu script preprocessing_augmentation.py
    mascara = criar_mascara_segmentacao_grabcut(img_cv2)
    
    if np.max(mascara) == 0: 
        mascara[:] = 255

    fundo_removido = cv2.bitwise_and(img_cv2, img_cv2, mask=mascara)
    img_realcada = aplicar_realce(fundo_removido)
    img_final_bgr = cv2.bitwise_and(img_realcada, img_realcada, mask=mascara)

    # 3. Converter de volta OpenCV (BGR) para PIL (RGB)
    img_final_rgb = cv2.cvtColor(img_final_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_final_rgb)

# --- 5. CLASSIFICAÇÃO (SEM THRESHOLD) ---
def classificar_comida(imagem):
    if imagem is None:
        return {}, None
    
    # PASSO A: Pré-processamento
    imagem_processada = executar_pipeline_blindado(imagem)
    
    # PASSO B: Previsão da IA
    img_tensor = transform(imagem_processada).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    # Retorna TODAS as probabilidades puras
    resultados = {}
    for i, prob in enumerate(probs):
        nome_classe = CLASS_NAMES[i]
        resultados[nome_classe] = float(prob)
    
    return resultados, imagem_processada

# --- 6. INTERFACE ---
interface = gr.Interface(
    fn=classificar_comida, 
    inputs=gr.Image(type="pil", label="Foto Original"), 
    outputs=[
        # O Gradio vai automaticamente ordenar e mostrar apenas os 5 maiores (num_top_classes)
        # mas o dicionário conterá tudo.
        gr.Label(num_top_classes=5, label="Probabilidades (Top 5)"),
        gr.Image(type="pil", label="Como a IA vê (Processada)")
    ],
    title="Detector de Alimentos - Pipeline Completo",
    description="Aplica segmentação GrabCut + Guilhotina Circular antes de classificar."
)

if __name__ == "__main__":
    interface.launch(inbrowser=True)