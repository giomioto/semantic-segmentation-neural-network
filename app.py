import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np # Adicionado para ler o arquivo de texto
import os

# --- 1. CONFIGURAÇÕES ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Alface', 'Almondega', 'Arroz', 'BatataFrita', 'Beterraba', 
               'BifeBovinoChapa', 'CarneBovinaPanela', 'Cenoura', 'FeijaoCarioca', 
               'Macarrao', 'Maionese', 'PeitoFrango', 'PureBatata', 
               'StrogonoffCarne', 'StrogonoffFrango', 'Tomate']
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = 'outputs_multilabel/best_ml.pth'
THRESHOLD_FILE = 'outputs_multilabel/best_thresholds.txt' # Caminho do arquivo gerado no notebook

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

# --- 3. CARREGAR THRESHOLDS (SEU TRECHO ADAPTADO) ---
try:
    if os.path.exists(THRESHOLD_FILE):
        print(f"Carregando thresholds de {THRESHOLD_FILE}...")
        th_values = np.loadtxt(THRESHOLD_FILE)
        # Mapeia cada valor para o nome da classe correspondente
        THRESHOLDS = {CLASS_NAMES[i]: float(th_values[i]) for i in range(len(CLASS_NAMES))}
        print("Thresholds personalizados carregados!")
        print(THRESHOLDS) # Print para você conferir no terminal
    else:
        raise FileNotFoundError
except Exception as e:
    print("⚠️ Aviso: arquivo de thresholds não encontrado. Usando padrão = 0.5")
    THRESHOLDS = {c: 0.5 for c in CLASS_NAMES}

# --- 4. TRANSFORMAÇÃO (Mantendo sua correção de Resize) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Sua correção fundamental para não cortar bordas
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 5. CLASSIFICAÇÃO COM FILTRO ---
def classificar_comida(imagem):
    if imagem is None:
        return {}
    
    img_tensor = transform(imagem).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    resultados = {}
    for i, prob in enumerate(probs):
        nome_classe = CLASS_NAMES[i]
        limite = THRESHOLDS.get(nome_classe, 0.5) # Pega o limite específico daquela comida
        
        # AQUI ESTÁ A MÁGICA:
        # Só adiciona ao resultado se a probabilidade vencer o limite daquela classe
        if prob >= limite:
            resultados[nome_classe] = float(prob)
    
    # Se o filtro for muito forte e não sobrar nada, retorna mensagem ou o top 1 bruto
    if not resultados:
        # Opcional: Retornar o item com maior probabilidade mesmo que seja baixa
        top_idx = np.argmax(probs)
        top_class = CLASS_NAMES[top_idx]
        resultados[top_class] = float(probs[top_idx])
        # Você pode adicionar um texto avisando que a confiança é baixa se quiser

    return resultados

# --- 6. INTERFACE ---
interface = gr.Interface(
    fn=classificar_comida, 
    inputs=gr.Image(type="pil", label="Arraste sua foto de comida aqui"), 
    outputs=gr.Label(num_top_classes=5, label="Alimentos Confirmados"),
    title="Detector de Alimentos com IA (Calibrado)",
    description="Detecta alimentos baseados em limiares de confiança específicos para cada classe."
)

if __name__ == "__main__":
    interface.launch(inbrowser=True)