import os
import shutil
import streamlit as st

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import create_retrieval_chain

from langgraph.graph import StateGraph, START, END

from typing import TypedDict


# =========================
# LOAD ENV
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# =========================
# STREAMLIT PAGE
# =========================

st.set_page_config(page_title="Simple RAG App")

st.title("📚 Simple RAG Application")


# =========================
# EXIT CONDITION
# =========================

exit_words = ["quit", "exit", "bye"]


# =========================
# CREATE FOLDERS
# =========================

os.makedirs("uploaded_docs", exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)


# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload Document",
    type=["pdf", "txt", "docx"]
)


# =========================
# LOAD DOCUMENT
# =========================

def load_document(file_path):

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)

    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)

    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)

    else:
        return []

    return loader.load()


# =========================
# CREATE VECTORSTORE
# =========================

def create_vectorstore(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(
        split_docs,
        embeddings
    )

    vectorstore.save_local("vectorstore")

    return vectorstore


# =========================
# LOAD VECTORSTORE
# =========================

def load_vectorstore():

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# =========================
# UPLOAD PROCESS
# =========================

if uploaded_file:

    file_path = os.path.join(
        "uploaded_docs",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("Uploaded File:", uploaded_file.name)
    st.write("Saved Path:", file_path)

    documents = load_document(file_path)

    create_vectorstore(documents)

    st.success("Document uploaded and embeddings created successfully!")


# =========================
# LLM
# =========================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    streaming=True,
    temperature=0
)


# =========================
# PROMPT
# =========================

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the provided context.

    <context>
    {context}
    </context>

    Question: {input}
    """
)


# =========================
# CREATE RAG CHAIN
# =========================

def get_rag_chain():

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain


# =========================
# LANGGRAPH STATE
# =========================

class GraphState(TypedDict):
    question: str
    answer: str


# =========================
# GRAPH NODE
# =========================

def rag_node(state: GraphState):

    question = state["question"]

    rag_chain = get_rag_chain()

    response = rag_chain.invoke({
        "input": question
    })

    return {
        "answer": response["answer"]
    }


# =========================
# BUILD LANGGRAPH
# =========================

builder = StateGraph(GraphState)

builder.add_node("rag_node", rag_node)

builder.add_edge(START, "rag_node")

builder.add_edge("rag_node", END)

graph = builder.compile()


# =========================
# USER INPUT
# =========================

user_question = st.chat_input(
    "Ask question from uploaded document..."
)


# =========================
# EXIT APPLICATION
# =========================

if user_question:

    if user_question.lower().strip() in exit_words:

        st.warning("Application Closed")

        st.stop()


# =========================
# RESPONSE
# =========================

if user_question:

    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        full_response = ""

        result = graph.invoke({
            "question": user_question
        })

        answer = result["answer"]

        # STREAMING EFFECT
        for word in answer.split():

            full_response += word + " "

            response_placeholder.markdown(full_response)

        st.write("")
