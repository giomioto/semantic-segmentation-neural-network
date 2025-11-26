import torch
import torchvision.transforms as T
from torchvision import models
from tkinter import Tk, filedialog
from PIL import Image
import json

# ================================
# CONFIGURAÇÕES
# ================================

MODEL_PATH = "outputs_multilabel/best_ml.pth"   # coloque o nome do seu modelo salvo
USE_THRESHOLDS = True              # True = usar thresholds otimizados
THRESHOLD_FILE = "outputs_multilabel/thresholds.json"  # se quiser salvar os thresholds

# Dicionário de classes (ajuste caso necessário)
class_to_idx = {
    "Alface": 0, "Almondega": 1, "Arroz": 2, "BatataFrita": 3, "Beterraba": 4,
    "BifeBovinoChapa": 5, "CarneBovinaPanela": 6, "Cenoura": 7, "FeijaoCarioca": 8,
    "Macarrao": 9, "Maionese": 10, "PeitoFrango": 11, "PureBatata": 12,
    "StrogonoffCarne": 13, "StrogonoffFrango": 14, "Tomate": 15
}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Se você salvou os thresholds em JSON
if USE_THRESHOLDS:
    try:
        with open(THRESHOLD_FILE, "r") as f:
            thresholds = json.load(f)
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

    results = []
    for i, p in enumerate(probs):
        cls = idx_to_class[i]
        th = thresholds.get(cls, 0.5)
        if p >= th:
            results.append((cls, float(p), float(th)))

    return results, probs

# ================================
# SELEÇÃO DA IMAGEM
# ================================
def pick_image():
    Tk().withdraw()  # esconde a janela principal
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

    classes, probs = predict_image(path)

    print("=== RESULTADOS ===")
    if len(classes) == 0:
        print("Nenhuma classe detectada.")
    else:
        for cls, p, th in classes:
            print(f"{cls}: prob={p:.3f}  |  threshold={th:.2f}")

    print("\n=== Probabilidades por classe ===")
    for i, p in enumerate(probs):
        print(f"{idx_to_class[i]}: {p:.3f}")
