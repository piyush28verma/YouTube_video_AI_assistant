# LLM-Powered YouTube Video Assistant (RAG)

A Retrieval-Augmented Generation (RAG) application built using **Python**, **LangChain**, **Streamlit**, **Google Gemini API**, and **FAISS**. It extracts transcripts from YouTube videos, generates text chunk embeddings, builds an in-memory vector index, and answers questions with context-aware responses.

---

## 🌟 Key Features

- **Automated Transcript Extraction**: Fetches YouTube transcripts with language fallback handling using `youtube-transcript-api`.
- **Text Chunking & Indexing**: Segments long transcripts using `RecursiveCharacterTextSplitter` into overlapping document chunks.
- **Gemini Embeddings**: Generates dense vector representations via `text-embedding-004`.
- **Semantic Vector Retrieval**: Performs fast similarity search using an in-memory **FAISS** vector store.
- **Context-Grounded Q&A**: Employs prompt engineering with `gemini-1.5-flash` to generate factual responses and minimize hallucinations.
- **Interactive Web Interface**: Streamlit UI with live transcript metrics, chat message history, and expandable transcript preview.

---

## 🏗️ Architecture

```
[ YouTube Video URL ] ──> [ transcript_loader.py ] ──> [ RecursiveCharacterTextSplitter ]
                                                                     │
                                                                     ▼
[ User Question ] ──> [ FAISS Similarity Search ] <── [ Gemini Embeddings (text-embedding-004) ]
                             │
                             ▼
              [ Context-Grounded Prompt Template ]
                             │
                             ▼
               [ ChatGoogleGenerativeAI (Gemini) ]
                             │
                             ▼
              [ Streamlit Interactive Chat UI ]
```

---

## 📁 Repository Structure

```
.
├── app.py                 # Streamlit web application & chat UI
├── rag_pipeline.py        # Text chunking, FAISS vector indexing & Gemini LLM chain
├── transcript_loader.py   # Transcript extraction & timestamp processing
├── requirements.txt       # Dependencies list
├── .env.example           # Environment template for GOOGLE_API_KEY
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone & Navigate
```bash
git clone <your-repository-url>
cd "LLM-Powered YouTube Video Assistant"
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```
