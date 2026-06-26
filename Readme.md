# AI-Assistant: Semantic Search over Video Transcripts

A Retrieval-Augmented Generation (RAG) project that enables semantic search over video transcripts. The project converts videos into text, generates embeddings using **BGE-M3** via **Ollama**, and retrieves the most relevant transcript chunks based on user queries using **cosine similarity**.

---

## 🚀 Features

* 🎥 Process video transcripts into searchable chunks
* ✂️ Intelligent text chunking
* 🧠 Generate embeddings using **BGE-M3**
* 🔍 Semantic search using cosine similarity
* 💬 Retrieve the most relevant transcript sections for user queries
* 📊 Store embeddings in a Pandas DataFrame

---

## 🛠️ Tech Stack

* Python
* Ollama
* BGE-M3 Embedding Model
* Pandas
* NumPy
* Scikit-learn
* Requests

---

## 📂 Project Structure

```text
video-rag/
│
├── jsons/
├── process_video.py
├── create_chunks.py
├── read_chunks.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ How It Works

```text
Video
   │
   ▼
Transcript Generation
   │
   ▼
Text Chunking
   │
   ▼
Generate BGE-M3 Embeddings
   │
   ▼
Store Embeddings
   │
   ▼
User Query
   │
   ▼
Query Embedding
   │
   ▼
Cosine Similarity Search
   │
   ▼
Top Relevant Chunks
```

---

## 📌 Installation

Clone the repository:

```bash
git clone https://github.com/Muskanchauhan29/AI-Assistant
cd AI-Assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Ollama:

```bash
ollama serve
```

Pull the embedding model:

```bash
ollama pull bge-m3
```

Run the project:

```bash
python read_chunks.py
```

---

## 💻 Example

```text
Ask a Question:
Where is machine learning introduced?

Searching...

Top Results:
Score: 0.91
Machine learning is a subset of artificial intelligence...

Score: 0.88
...
```

---

## 🔮 Future Improvements

* Build a web interface using Streamlit or Gradio
* Support multiple videos and document collections
* Add metadata filtering for improved retrieval

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome! Feel free to fork the repository and submit a pull request.

---

