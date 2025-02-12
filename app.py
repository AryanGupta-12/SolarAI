import faiss
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
import gradio as gr
import numpy as np
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("solar_vectors.index")
with open("chunks.pkl", "rb") as f:
    chunks= pickle.load(f)

os.environ["GROQ_API_KEY"]= "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"
llm = ChatGroq(model="mixtral-8x7b-32768",temperature=0)

def retrieve_relevant_text(query, top_k=1):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

def generate_response(user_query):
    retrieved_text = retrieve_relevant_text(user_query, top_k=4)
    system_message = "You are an intelligent assistant that provides accurate, helpful information about solar energy based on the information provided(if not, answer according to your knowledge)."
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", f"Use the following information to answer: {retrieved_text} \n\nUser Query: {user_query}")
    ])

    chain = prompt_template | llm
    response = chain.invoke({"text": user_query})
    return response.content

def gradio_chatbot(user_input):
    response = generate_response(user_input)
    return response



with gr.Blocks() as demo:
    gr.Markdown("# 🌞 SolarAI 🌞")

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Ask me anything about solar energy...",
            lines=2,
            interactive=True
        )

    with gr.Row():
        output_box = gr.Textbox(
            lines=12,
            interactive=True,
            label="Chatbot Response"
        )

    submit_btn = gr.Button("Ask")
    submit_btn.click(fn=gradio_chatbot, inputs=user_input, outputs=output_box)
demo.launch()