import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2 
from collections import defaultdict
import operator
import io

# --- IMPORTANDO SEU SCRIPT DE PRÉ-PROCESSAMENTO ---
# O usuário deve garantir que este arquivo esteja presente
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

# --- 1. CONFIGURAÇÕES ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ATENÇÃO: Verifique se esta lista de classes está correta para o seu modelo best_single.pth
CLASS_NAMES = ['Alface', 'Almondega', 'Arroz', 'BatataFrita', 'Beterraba', 
               'BifeBovinoChapa', 'CarneBovinaPanela', 'Cenoura', 'FeijaoCarioca', 
               'Macarrao', 'Maionese', 'PeitoFrango', 'PureBatata', 
               'StrogonoffCarne', 'StrogonoffFrango', 'Tomate']
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH_CLASSIFICATION = 'outputs_multilabel/best_single.pth' # Caminho do modelo de classificação do usuário

# Limite mínimo de confiança para a classificação
CLASSIFICATION_THRESHOLD = 0.60 

# Parâmetros de PDI
MIN_CONTOUR_AREA = 500 # Área mínima do contorno para ser considerado um alimento

# --- 2. CARREGAR MODELO SINGLE-LABEL (CLASSIFICAÇÃO) ---
def get_classification_model():
    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, NUM_CLASSES)
    return model

print("Carregando modelo SINGLE-LABEL (Classificação)...")
classification_model = get_classification_model()
try:
    classification_model.load_state_dict(torch.load(MODEL_PATH_CLASSIFICATION, map_location=DEVICE))
    print("Modelo single-label carregado com sucesso!")
except FileNotFoundError:
    print(f"ERRO CRÍTICO: Não achei o arquivo {MODEL_PATH_CLASSIFICATION}. O modelo não funcionará corretamente.")

classification_model.to(DEVICE)
classification_model.eval()

# --- 3. TRANSFORMAÇÃO FINAL (CLASSIFICAÇÃO) ---
classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. FUNÇÃO DE PRÉ-PROCESSAMENTO SIMPLIFICADA (COM RETORNO DA MÁSCARA) ---
def aplicar_remocao_fundo(imagem_pil):
    # Converte PIL Image para numpy array (BGR para OpenCV)
    img_np = np.array(imagem_pil)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 1. Segmentação (usando a função do usuário)
    mascara = criar_mascara_segmentacao_grabcut(img_cv2)
    
    # Garante que a máscara não seja totalmente preta se o GrabCut falhar
    if np.max(mascara) == 0:
        mascara[:] = 255

    # Aplica a máscara para remover o fundo
    fundo_removido = cv2.bitwise_and(img_cv2, img_cv2, mask=mascara)

    # 2. Converte de volta para RGB
    img_final_rgb = cv2.cvtColor(fundo_removido, cv2.COLOR_BGR2RGB)

    # Retorna a imagem processada (PIL) e a máscara (numpy)
    return Image.fromarray(img_final_rgb), mascara

# --- 5. CLASSIFICAÇÃO DE UM ÚNICO RECORTE ---
def classify_single_crop(imagem_pil):
    if imagem_pil is None:
        return "N/A", 0.0
    
    # O recorte é classificado diretamente, sem PDI extra
    imagem_processada = imagem_pil 
    
    img_tensor = classification_transform(imagem_processada).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = classification_model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_index = np.argmax(probs)
    top_class = CLASS_NAMES[top_index]
    top_confidence = float(probs[top_index])

    return top_class, top_confidence

# --- 6. FUNÇÃO PRINCIPAL: ABORDAGEM SIMPLES DE GRADE 3X3 (3 SAÍDAS) ---
def classificar_comida(imagem_pil):
    if imagem_pil is None:
        return {}, Image.new('RGB', (224, 224), color = 'black'), {"status": "Erro: Imagem vazia."}

    # Aplica a remoção de fundo na imagem inteira e obtém a máscara
    imagem_processada_pil, mascara_grabcut = aplicar_remocao_fundo(imagem_pil)
    img_np = np.array(imagem_processada_pil) # Imagem processada em numpy (RGB)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # Converte para BGR para OpenCV
    
    # Obtém as dimensões da imagem
    img_h, img_w, _ = img_cv2.shape

    # --- 6.1 Encontrar o Contorno Principal (Prato) ---
    
    # Encontra os contornos na máscara do GrabCut
    contours, _ = cv2.findContours(mascara_grabcut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {}, Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)), {"status": "Erro: Não foi possível encontrar o contorno principal na máscara do GrabCut."}

    # Assume que o maior contorno é o prato
    main_contour = max(contours, key=cv2.contourArea)
    
    # Obtém a caixa delimitadora do prato
    x_main, y_main, w_main, h_main = cv2.boundingRect(main_contour)
    
    # --- 6.2 Estratégia de Divisão Simples (Grade 3x3) ---
    
    bboxes_to_classify = []
    
    # Função auxiliar para gerar grades
    def generate_grid_bboxes(cols, rows):
        grid_bboxes = []
        col_step = w_main / cols
        row_step = h_main / rows
        for i in range(cols):
            for j in range(rows):
                x1 = int(x_main + i * col_step)
                y1 = int(y_main + j * row_step)
                x2 = int(x_main + (i + 1) * col_step)
                y2 = int(y_main + (j + 1) * row_step)
                grid_bboxes.append((x1, y1, x2, y2))
        return grid_bboxes

    # Apenas a Grade 3x3 (9 recortes)
    bboxes_to_classify.extend(generate_grid_bboxes(3, 3))

    # Remove caixas muito pequenas
    unique_bboxes = []
    for bbox in bboxes_to_classify:
        x1, y1, x2, y2 = bbox
        # Garante que a área do recorte não seja muito pequena
        if (x2 - x1) * (y2 - y1) > MIN_CONTOUR_AREA and bbox not in unique_bboxes:
            unique_bboxes.append(bbox)

    all_results_json = []
    consolidated_results_label = defaultdict(lambda: {"count": 0, "total_confidence": 0.0})
    
    # --- 6.3 Classificação e Desenho ---
    
    for x1, y1, x2, y2 in unique_bboxes:
        
        # Recorta a imagem original
        cropped_img = imagem_pil.crop((x1, y1, x2, y2))
        
        # Classifica o recorte
        top_class, confidence = classify_single_crop(cropped_img)
        
        # Armazena o resultado para a saída JSON e desenha a caixa (sem filtro de confiança)
        all_results_json.append({
            "bounding_box": [x1, y1, x2, y2],
            "classified_food": top_class, 
            "confidence": f"{confidence:.4f}"
        })
        
        # Desenha a caixa e o rótulo na imagem processada para visualização
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv2, f"{top_class} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Consolida o resultado para a saída Label SOMENTE se a confiança for alta
        if confidence >= CLASSIFICATION_THRESHOLD:
            consolidated_results_label[top_class]["count"] += 1
            consolidated_results_label[top_class]["total_confidence"] += confidence

    # --- 7. Prepara as 3 saídas para a interface Gradio original ---

    # Saída 1: gr.Label (Alimentos Detectados Consolidado)
    resultados_label = {}
    for food, data in consolidated_results_label.items():
        # Calcula a confiança média
        resultados_label[food] = data["total_confidence"] / data["count"]
    
    # Saída 2: gr.Image (Como a IA vê - Processada)
    img_with_boxes = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    
    # Saída 3: gr.JSON (Detalhes da Detecção)
    if not all_results_json:
        all_results_json.append({"status": f"Nenhuma detecção de objeto encontrada com confiança > {CLASSIFICATION_THRESHOLD}."})
        
    resultados_detec = {
        "Alimentos Detectados": all_results_json
    }

    # Retorna os 3 valores na ordem correta: Label, Imagem, JSON
    return resultados_label, img_with_boxes, resultados_detec

# --- 8. INTERFACE GRADIO (ORIGINAL DO USUÁRIO REINTRODUZIDA) ---
interface = gr.Interface(
    fn=classificar_comida, 
    inputs=gr.Image(type="pil", label="Foto Original"), 
    outputs=[
        gr.Label(num_top_classes=5, label="Alimentos Detectados (Consolidado)"),
        gr.Image(type="pil", label="Como a IA vê (Processada)"),
        gr.JSON(label="Detalhes da Detecção (Top 1 de cada recorte)")
    ],
    title="Classificador de Alimentos por Grade 3x3 Fixa + ResNet",
    description=(
        f"Esta interface usa a máscara do GrabCut para encontrar o prato e o divide em uma grade 3x3 fixa para classificação. "
        f"Apenas resultados de classificação com confiança > {CLASSIFICATION_THRESHOLD} são exibidos."
    )
)

if __name__ == "__main__":
    interface.launch(inbrowser=True)
