# Multilingual Document Summarizer (RAG)

A Streamlit-based application that summarizes documents using Retrieval-Augmented Generation (RAG) and supports multilingual output using Google Gemini. 

This project supports summarization of PDF, DOCX, and TXT files using either **Extractive** or **Abstractive** methods and can translate summaries to languages like Hindi, Tamil, and Marathi.

---

## ✨ Features

- 📄 Supports multiple file formats (PDF, DOCX, TXT)
- 🔍 Extractive and Abstractive summarization using LLMs
- 🌍 Language detection and translation support
- 🔌 Google Gemini-based summarization and translation
- 🧠 Embedding-based RAG using custom HuggingFace models

---

## 🏗 Project Structure

core/
│
├── document_processor.py       # Handles file loading, text extraction, chunking
├── rag_summarizer.py           # RAG pipeline: embeddings, vectorstore, LLM summary
└── translation_manager.py      # Translates summaries into user-selected languages

app.py                          # Streamlit frontend

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/document-summarizer.git
cd document-summarizer
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

Create a `.env` file in the root directory:

```
GOOGLE_API_KEY=your_google_api_key_here
HF_TOKEN=your_hf_token
```

---

## 🧠 Model Details

- **Embeddings**: `nomic-ai/nomic-embed-text-v2-moe` (HuggingFace)
- **LLM**: Google Gemini (`gemini-2.0-flash` by default)
- **Vector DB**: In-memory Chroma via Langchain

---

## 🖥 Usage

Start the Streamlit app:

```bash
streamlit run app.py
```

- Upload a document (`.pdf`, `.docx`, `.txt`) or paste text.
- Choose summarization type: `Extractive` or `Abstractive`.
- Optionally select a target language for translated summary.
- Click **Summarize** to generate and view results.

---

## 📦 Example Output

**Original Text:**
> This document outlines the core architecture of the summarizer application...

**Summary (Abstractive):**
> The app uses RAG with Gemini and Nomic embeddings to summarize and translate documents interactively.

**Translated (Hindi):**
> यह ऐप दस्तावेज़ों को इंटरैक्टिव रूप से सारांशित और अनुवाद करने के लिए RAG और Gemini का उपयोग करता है।
