# 🌞 SolarAI Chatbot

This project is an **AI-powered chatbot** that provides accurate and insightful information about the **solar industry**, including **solar panel technology, installation processes, maintenance, costs, ROI analysis, and market trends**. The chatbot integrates **LLM (ChatGroq - Mixtral-8x7B)** with **vector search (FAISS)** for better context-aware responses.
---
## Table of Contents
**1.Features**

**2.Installation & Setup**

**3.Code Breakdown (Function-by-Function)**

**4.Example Convesations**

**5.Deployment Guide**

---
## 📌 Features

- Extracts solar panels knowledge from a **DOCX file** (manually created from data available on internet)
- Converts text into **embeddings** using `SentenceTransformer`
- Stores embeddings in a **FAISS vector database** for efficient retrieval
- Queries relevant information before sending it to **ChatGroq (Mixtral-8x7B)**
- Provides an **interactive chatbot UI using Gradio**

---

## 🛠 Installation & Setup

### **Step 1: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 2: Run the Chatbot**

```bash
python app.py
```

---

## 📂 Code Breakdown (Function-by-Function)

### **1️⃣ Extracting Text from DOCX**

```python
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text
```

🔹 **Purpose:** Reads a `.docx` file and extracts useful solar-related information.

---

### **2️⃣ Splitting Text into Chunks**

```python
def split_text(text, chunk_size=300):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
```

🔹 **Purpose:** Splits large text data into smaller, meaningful **chunks** for better vector search performance.

---

### **3️⃣ Generating Embeddings**

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")  # Embedding model
embeddings = model.encode(chunks)
```

🔹 **Purpose:** Converts text **chunks** into numerical representations (vectors) for similarity search.

---

### **4️⃣ Storing Embeddings in FAISS Vector Database**

```python
import faiss
import numpy as np
vector_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(vector_dim)
index.add(np.array(embeddings))
faiss.write_index(index, "solar_vectors.index")
```

🔹 **Purpose:** Uses **FAISS** to efficiently store and retrieve relevant text when a user asks a question.

---

### **5️⃣ Retrieving Relevant Information**

```python
def retrieve_relevant_text(query, top_k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return " ".join([chunks[i] for i in indices[0]])
```

🔹 **Purpose:** Finds the **most relevant** pieces of information to **pass to the chatbot** before generating a response.

---

### **6️⃣ Chatbot Integration with ChatGroq (Mixtral-8x7B)**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.2)
def chat_with_groq(user_query):
    retrieved_text = retrieve_relevant_text(user_query)
    system_message = "You are an AI assistant that provides accurate solar energy information."
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", f"Use the following information to answer: {retrieved_text} \n\nUser Query: {user_query}")
    ])
    chain = prompt_template | llm
    response = chain.invoke({"text": user_query})
    return response.content
```

🔹 **Purpose:** Uses **retrieved data + user query** to generate an **LLM-based response**.

---

### **7️⃣ Gradio Chatbot UI**

```python
import gradio as gr
def gradio_chatbot(user_input):
    response = chat_with_groq(user_input)
    return response
with gr.Blocks() as demo:
    gr.Markdown("# 🌞 SolarAI 🌞")
    with gr.Row():
        user_input = gr.Textbox(placeholder="Ask me anything about solar energy...", lines=2, interactive=True)
    with gr.Row():
        output_box = gr.Textbox(lines=6, interactive=True, label="Chatbot Response")
    submit_btn = gr.Button("Ask")
    submit_btn.click(fn=gradio_chatbot, inputs=user_input, outputs=output_box)
demo.launch()
```

🔹 **Purpose:** Creates a **Gradio-powered UI** for user interaction with the chatbot.

---
## Example Convesations:

![Image](https://github.com/user-attachments/assets/187c1dcd-907d-4ed9-9f37-743848baf462)

![Image](https://github.com/user-attachments/assets/96329dfb-6fdc-4f78-b405-5ec863de893d)

---
## 🚀 Deployment Guide

### **Option 1: Run Locally**

```bash
python app.py
```

### **Option 2: Deploy on Hugging Face Spaces**

1. Create `requirements.txt`.

2. Push to Hugging Face:

```bash
git init
git add .
git commit -m "Deploy Solar Chatbot"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/solar-chatbot
git push origin main
```

✅ Your chatbot is now **LIVE** on Hugging Face Spaces!

---

## 🎯 Future Improvements

✅ Add **voice-based interaction** 🎙️ ✅ Improve **multi-turn conversation memory** ✅ Enable **real-time solar industry data fetching** ✅ Integrate **WhatsApp/Telegram bot support** 📲
