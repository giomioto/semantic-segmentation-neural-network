# Semantic Segmentation Neural Network (Food Classification -- Single & Multi-Label)

Este repositório refere-se ao trabalho final da disciplina de
Processamento Digital de Imagens, do Departamento de Informática da
Universidade Tecnológica Federal do Paraná (UTFPR).

![UTFPR](https://utfpr-ct-static-content.s3.amazonaws.com/utfpr.curitiba.br/wp-content/uploads/sites/7/2019/11/utfpr1.png)

## 📌 Descrição do Projeto

Este projeto implementa um pipeline completo de visão computacional para
classificação de alimentos usando CNNs ResNet.

### 1. Classificação Single-Label

-   Baseada na pasta `Imagens_um_Alimento`
-   Treinamento com CrossEntropyLoss
-   Uma classe por imagem

### 2. Classificação Multi-Label

-   Baseada na pasta `Imagens_Varios_Alimentos`
-   Saída multi-hot
-   Loss: BCEWithLogitsLoss
-   Métricas: F1, Hamming Loss
-   Threshold otimizado por classe

## 📁 Estrutura do Projeto

    ├── Imagens_um_Alimento/
    ├── Imagens_Varios_Alimentos/
    ├── notebooks/
    │   └── treino_pipeline.ipynb
    ├── scripts/
    │   ├── train_multilabel.py
    │   ├── predict_image.py
    ├── outputs_single/
    ├── outputs_multilabel/
    └── README.md

## 🚀 Como executar

Clone:

    git clone <repo>

Instale dependências:

    pip install -r requirements.txt

Execute o notebook:

    notebooks/treino_pipeline.ipynb

Ou o script:

    python scripts/train_multilabel.py

## 🔍 Previsão manual

Selecionar imagem pelo explorador:

    python scripts/predict_image.py

## 📊 Resultados

### Single-label

-   Test Accuracy ≈ 99.5%

### Multi-label

-   Micro-F1 ≈ 0.96
-   Hamming Loss ≈ 0.0054

## 🧑‍🏫 Autores

Trabalho final da disciplina de Processamento Digital de Imagens --
UTFPR.

## 📜 Licença

MIT License.
