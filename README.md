# Emotional Manipulation Detector

**CB011219 – Lasith Siriwardena**  
Final Year Project — APIIT / University of Staffordshire  
Supervisor: Mr. Janotheepan Mariyathas

---

## Overview

An AI-powered web application that detects emotional manipulation in online text-based conversations. The system uses a fine-tuned BERT transformer model to classify conversations into seven categories:

- Neutral
- Guilt Tripping
- Gaslighting
- Love Bombing
- Direct Coercion
- Charm / Flattery
- Passive Aggressive

The application supports two input methods:
1. **Manual text input** — type or paste a conversation directly
2. **Screenshot upload** — upload a chat screenshot and the system extracts text via OCR

An explainable AI (XAI) component highlights the words that most influenced the prediction using BERT attention weights.

---

## Project Structure

```
fyp_app/
│
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── model/                        # Trained BERT model files
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── label_encoder.pkl
│
├── Google_collab_notebooks/
│   ├── FYP_Training_Notebook.ipynb     # Original training notebook
│   ├──
│   └── FYP_Demo_Notebook.ipynb         # Clean demo notebook
│
└── =
```

---

## Requirements

### System
- Python 3.11 or higher
- Tesseract OCR 5.x (for screenshot analysis)
- macOS, Windows, or Linux

### Install Tesseract

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/Lxith2007fyp-manipulation-detector.git
cd fyp-manipulation-detector
```

**2. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the model files**

Download `fyp_bert_model.zip` and extract it into the `model/` folder:
```bash
unzip fyp_bert_model.zip -d model
```

---

## Running the Application

```bash
cd fyp_app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## Training the Model

Open `notebooks/FYP_Training_Notebook.ipynb` in Google Colab:

1. Set runtime to **T4 GPU** (Runtime → Change runtime type)
2. Run all cells
3. Upload `archive.zip` when prompted (the training dataset)
4. Download `fyp_bert_model.zip` at the end

**Training details:**
- Model: bert-base-uncased
- Dataset: 10,000 labelled manipulation conversations
- Split: 70% train / 15% validation / 15% test
- Epochs: 3
- Batch size: 16
- Learning rate: 2e-5
- GPU: NVIDIA Tesla T4

---

## Results

### Primary Dataset
| Metric | Score |
|--------|-------|
| Accuracy | 99.93% |
| Precision (macro avg) | 1.00 |
| Recall (macro avg) | 1.00 |
| F1-score (macro avg) | 1.00 |

### Cross-Dataset Evaluation (MentalManip)
| Metric | Score |
|--------|-------|
| Binary Accuracy | 70.77% |
| Manipulative Recall | 99% |

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.11 |
| ML Framework | PyTorch |
| NLP Model | BERT (bert-base-uncased) |
| NLP Library | Hugging Face Transformers |
| Web Interface | Streamlit |
| OCR Engine | Tesseract 5.5.3 |
| OCR Interface | pytesseract |
| Image Processing | Pillow |
| Evaluation | Scikit-learn |
| Training Environment | Google Colaboratory (T4 GPU) |

---

## Datasets

- **Primary:** Purpose-built manipulation conversation dataset (10,000 entries, 7 classes, JSONL format)
- **Secondary:** MentalManip — Wang et al. (2024), 4,000 annotated movie dialogues

---

## Limitations

- Inference runs on CPU locally (~1.5–2 seconds per prediction). On GPU this drops to 11–39ms.
- OCR accuracy varies across chat platforms and image quality. The editable text box allows manual correction.
- The model may misclassify very short or subtly manipulative exchanges as neutral.
- Performance on real-world data (70.77% on MentalManip) is lower than on structured training data (99.93%), reflecting the domain gap between synthetic training conversations and organic dialogue.

