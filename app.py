import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import tempfile
import os

st.set_page_config(page_title="Asistente Legal Peruano", page_icon="⚖️")

st.title("⚖️ Asistente Legal Peruano")
st.markdown("Sube un documento legal (PDF) y hazle preguntas en lenguaje natural.")

# API KEY
openai_api_key = st.text_input("🔑 Ingresa tu clave API de OpenAI", type="password")

# Cargar archivo PDF
pdf_file = st.file_uploader("📄 Carga un documento legal (PDF)", type=["pdf"])

if pdf_file and openai_api_key:
    with st.spinner("Procesando documento..."):
        # Guardar PDF temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_pdf_path = tmp_file.name

        # Leer y dividir PDF
        loader = PyPDFLoader(tmp_pdf_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(pages)

        # Embeddings y FAISS
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever()
        llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("✅ Documento cargado y listo para consultas.")

        # Pregunta del usuario
        query = st.text_input("🗣️ Escribe tu pregunta legal:")

        if query:
            with st.spinner("Buscando respuesta..."):
                respuesta = qa_chain.run(query)
                st.markdown(f"### 📜 Respuesta:")
                st.write(respuesta)
