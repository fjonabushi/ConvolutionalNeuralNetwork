# Cell Image Classification Using Convolutional Neural Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red) ![Status](https://img.shields.io/badge/Status-Research_Thesis-yellow)

This project was submitted as my **Master's thesis at Epoka University** and explores **deep learning methods for classifying healthy vs. unhealthy cell images** using a modified **LeNet architecture**.

---

## 🩺 Project Overview
- **Objective:** Predict healthy vs. unhealthy cell images using Convolutional Neural Networks.
- **Datasets:** Three datasets containing **9,332, 20,102, and 12,520 images** were used, trained and evaluated separately.
- **Preprocessing Methods:**
  - Unsharp masking
  - Median filtering
  - High-pass filtering
- **Architecture:** Modified LeNet CNN for grayscale and color cell images.

---

## 🚀 Features
✅ Modified LeNet CNN architecture with additional layers for stability and accuracy.  
✅ Preprocessing pipeline for medical image preparation.  
✅ Modular, clean code with comments for reproducibility.  
✅ Configurable training and testing pipeline for different datasets.

---

## ⚠️ Note
This repository currently contains **only the source code and final submited thesis**. Due to size and licensing constraints, **datasets are not included**.

---

## 🛠️ Installation & Usage

1️⃣ Clone the repository:
```bash
git clone https://github.com/fjonabushi/ConvolutionalNeuralNetwork.git
cd ConvolutionalNeuralNetwork
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

3️⃣ Run training:
```bash
python train.py
```


---

## 📁 Repository Structure
```
ConvolutionalNeuralNetwork/
│
├── data/               # Data preprocessing
├── models/             # Saved model checkpoints
├── src/                # Core training and model files
│   ├── model.py
│   ├── dataset.py
│   └── train.py
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

---

## 👩‍💻 Author
**Fjona Bushi**  
[LinkedIn](https://www.linkedin.com/in/fjona-h-84213a190) | [Email](mailto:fiona725f@gmail.com)

---

*Always exploring and building with curiosity.*
