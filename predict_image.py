import torch
import torchvision.transforms as T
from torchvision import models
from tkinter import Tk, filedialog
from PIL import Image
import numpy as np
import os

# ================================
# CONFIGURAÇÕES
# ================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # pasta do script
MODEL_PATH = os.path.join(BASE_DIR, "best_ml.pth")
THRESHOLD_FILE = os.path.join(BASE_DIR, "best_thresholds.txt")
USE_THRESHOLDS = True
class_to_idx = {
    "Alface": 0, "Almondega": 1, "Arroz": 2, "BatataFrita": 3, "Beterraba": 4,
    "BifeBovinoChapa": 5, "CarneBovinaPanela": 6, "Cenoura": 7, "FeijaoCarioca": 8,
    "Macarrao": 9, "Maionese": 10, "PeitoFrango": 11, "PureBatata": 12,
    "StrogonoffCarne": 13, "StrogonoffFrango": 14, "Tomate": 15
}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# ================================
# LEITURA DE THRESHOLDS
# ================================
if USE_THRESHOLDS:
    try:
        th_values = np.loadtxt(THRESHOLD_FILE)
        thresholds = {cls: float(th_values[i]) for i, cls in enumerate(class_to_idx)}
    except:
        print("⚠️ Aviso: arquivo de thresholds não encontrado. Usando threshold padrão = 0.5")
        thresholds = {c: 0.5 for c in class_to_idx.keys()}
else:
    thresholds = {c: 0.5 for c in class_to_idx.keys()}

# ================================
# CARREGA MODELO
# ================================
num_classes = len(class_to_idx)
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ================================
# TRANSFORM
# ================================
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ================================
# FUNÇÃO DE PREDIÇÃO
# ================================
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].numpy()

    # aplica thresholds
    results = []
    for i, p in enumerate(probs):
        cls = idx_to_class[i]
        th = thresholds.get(cls, 0.5)
        if p >= th:
            results.append((cls, float(p), float(th)))

    # top-3 mesmo abaixo do threshold
    top_idx = probs.argsort()[::-1][:3]
    top3 = [(idx_to_class[i], float(probs[i])) for i in top_idx]

    return results, probs, top3

# ================================
# SELEÇÃO DA IMAGEM
# ================================
def pick_image():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Selecione uma imagem",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png")]
    )
    return file_path

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    print("📁 Escolha uma imagem para predizer...")
    path = pick_image()

    if not path:
        print("Nenhuma imagem selecionada.")
        exit()

    print(f"\n🔍 Imagem selecionada: {path}\n")

    classes, probs, top3 = predict_image(path)

    print("=== RESULTADOS COM THRESHOLD ===")
    if len(classes) == 0:
        print("Nenhuma classe detectada acima do threshold.")
    else:
        for cls, p, th in classes:
            print(f"{cls}: prob={p:.3f}  |  threshold={th:.2f}")

    print("\n=== TOP-3 CLASSES MESMO ABAIXO DO THRESHOLD ===")
    for cls, p in top3:
        print(f"{cls}: prob={p:.3f}")

    print("\n=== Probabilidades por classe ===")
    for i, p in enumerate(probs):
        print(f"{idx_to_class[i]}: {p:.3f}")
