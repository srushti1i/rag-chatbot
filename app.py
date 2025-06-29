import streamlit as st
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch


st.set_page_config(page_title="PDF Question Answering", layout="centered")

# Title
st.title("📄 Ask Questions About Your PDF")

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

@st.cache_data(show_spinner=False)
def extract_and_clean_text(uploaded_file):
    # Extract
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    # Clean
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

@st.cache_resource(show_spinner=False)
def setup_model_and_embeddings(cleaned_text):
    # 1. Split text into clean overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(cleaned_text)

    # 2. Remove junk/short chunks
    texts = [t for t in texts if len(t.strip()) > 50]

    # 3. Generate embeddings
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = embedder.encode(texts)

    # Use LangChain's FAISS wrapper
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts, embedding_model)

    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})


    # 5. Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)

    return texts, retriever, tokenizer, model, device


if pdf_file:
    with st.spinner("Processing PDF..."):
        raw_text = extract_and_clean_text(pdf_file)
        texts, retriever, tokenizer, model, device = setup_model_and_embeddings(raw_text)
        st.success("PDF processed. You can now ask questions!")

    # Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Ask a question about the document:")

if st.button("Submit") and question:
    with st.spinner("Generating answer..."):
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs])


        #prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        prompt = f"You are a helpful AI assistant. Use only the information from the context to answer the question clearly.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

        #inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        outputs = model.generate(**inputs, max_length=150)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if len(answer.strip()) < 5 or "tactile" in answer.lower():
            answer = "I'm sorry, I couldn't find a clear answer in the document."

        # Save to chat history
        st.session_state.chat_history.append((question, answer))

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 💬 Chat History")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**AI:** {a}")
        st.markdown("---")

