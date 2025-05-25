# 🩺 CheXpert + LLM-Powered X-ray Report Generator

This project combines deep learning-based **multi-label classification** of chest X-rays with **LLM-driven report generation**, using a hybrid architecture built on PyTorch, LangChain, and Chroma vector stores.

---

## 🧠 Overview

### Workflow:
1. 🖼️ **Input**: Chest X-ray image  
2. 🤖 **Step 1**: CNN-based classifier (DenseNet) predicts medical tags (e.g., Cardiomegaly, Edema)  
3. 📚 **Step 2**: Predicted tags are used to **query a Chroma vector store** built from the IU Chest X-ray dataset  
4. 💬 **Step 3**: Retrieved context documents are passed to an **LLM** (e.g., DeepSeek Chat via OpenRouter) to generate a comprehensive **radiology report**

---

## 🗃️ Dataset

- **Classification model** trained on the [CheXpert dataset]((https://www.kaggle.com/datasets/mimsadiislam/chexpert))
- **Report retrieval** uses embeddings generated from the **IU Chest X-ray dataset** (open-access reports)

---

## 🛠️ Tech Stack

- **PyTorch**: For multi-label image classification (DenseNet)
- **LangChain**: Framework for chaining together LLM operations
- **ChromaDB**: Vector database for storing and retrieving medical text embeddings
- **SentenceTransformers**: For generating document embeddings
- **OpenRouter + DeepSeek LLM**: For contextual report generation

---

## 🧪 Model Architecture

### 🧬 Classification:
- **Backbone**: Pretrained DenseNet (DenseNet121)
- **Loss Function**: Weighted BCEWithLogitsLoss
- **Output**: 14 disease-specific logits

### 🧠 LLM Component:
- Uses LangChain's `ChatOpenAI` to call `deepseek-chat`
- Tags are used as a query to ChromaDB
- Top-k documents are retrieved and passed as context to the LLM
- LLM returns a **free-text medical report**

---

## 📊 Metrics

- **Classification**:
  - AUC, F1, Precision, Recall for each tag
- **LLM Output**:
  - Qualitatively evaluated on correctness and coherence

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install torch torchvision langchain-community chromadb sentence-transformers openai
