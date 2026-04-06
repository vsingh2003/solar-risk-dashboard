"""
RAG-Based Chatbot Page
Reads the final_anomaly_results.csv and allows users to ask questions.
"""
import streamlit as st
import os
import pandas as pd
import tempfile

# ── RAG Dependencies ──
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG & STYLING (Matching Main App)
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Solar AI Assistant", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #161b22 0%, #0d1117 100%); }
    h1, h2, h3 { color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Solar AI Assistant (RAG)")
st.caption("Ask natural language questions about the anomalies, units, and power generation.")
st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  API KEY SETUP
# ══════════════════════════════════════════════════════════════
# Ask the user for their API key securely in the sidebar
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
    st.caption("Required to power the Retrieval-Augmented Generation (RAG) pipeline.")

if not openai_api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to activate the chatbot.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# ══════════════════════════════════════════════════════════════
#  RAG PIPELINE: LOAD DATA & CREATE VECTOR STORE
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Building Vector Database...")
def build_vector_store():
    # 1. Load the dataset
    dataset_path = "final_anomaly_results.csv"
    if not os.path.exists(dataset_path):
        st.error("Dataset not found. Please run the pipeline script first.")
        st.stop()
        
    # 2. Use Langchain's CSV Loader
    loader = CSVLoader(file_path=dataset_path, encoding="utf-8")
    documents = loader.load()
    
    # 3. Create Embeddings and Store in FAISS Vector DB
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Initialize Vector Database
vectorstore = build_vector_store()

# Initialize LLM and QA Chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 most relevant rows
)

# ══════════════════════════════════════════════════════════════
#  CHAT INTERFACE
# ══════════════════════════════════════════════════════════════
# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I have read the `final_anomaly_results.csv` data. What would you like to know about the solar anomalies?"}
    ]

# Display historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input Box
if prompt := st.chat_input("e.g., Which unit had the sharpest drop in power?"):
    
    # 1. Add user message to UI and session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get AI response via RAG
    with st.chat_message("assistant"):
        with st.spinner("Searching the data..."):
            try:
                response = qa_chain.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error querying the AI: {str(e)}")