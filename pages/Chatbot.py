"""
RAG-Based Chatbot Page — Solar AI Assistant
Zero dependency on the bare `langchain` package.
Uses only langchain_core, langchain_community, langchain_openai.
Compatible with Streamlit Cloud (Python 3.12) and all modern LangChain versions.
"""
import os
import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG & STYLING
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Solar AI Assistant", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
    h1, h2, h3 { color: #58a6ff !important; }
    .stChatMessage { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Solar AI Assistant (RAG)")
st.caption("Ask natural language questions about the anomalies, units, and power generation data.")
st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  SIDEBAR — API KEY
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
    st.caption("Required to power the RAG pipeline.")
    st.markdown("---")
    st.markdown("**Example questions:**")
    st.caption("• Which unit had the most anomalies?")
    st.caption("• What was the sharpest single-day drop?")
    st.caption("• List all anomalies for unit INV-01.")
    st.caption("• What does a negative z-score mean here?")

if not openai_api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to activate the chatbot.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# ══════════════════════════════════════════════════════════════
#  RAG PIPELINE — PURE langchain_core LCEL CHAIN
# ══════════════════════════════════════════════════════════════
def format_docs(docs):
    """Concatenate retrieved document page_content into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource(show_spinner="📚 Building vector database from CSV...")
def build_rag_chain(api_key: str):
    """
    Builds a fully LCEL-native RAG chain using zero imports from
    the bare `langchain` package. All imports are from langchain_core
    and langchain_community which are stable across all recent versions.
    """
    dataset_path = "final_anomaly_results.csv"
    if not os.path.exists(dataset_path):
        return None

    # 1. Load CSV rows as Documents
    loader    = CSVLoader(file_path=dataset_path, encoding="utf-8")
    documents = loader.load()

    # 2. Embed into FAISS vector store
    embeddings  = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 3. Prompt template
    prompt = ChatPromptTemplate.from_template("""
You are a solar energy data analyst assistant.
Answer the user's question using ONLY the context rows provided below.
Be concise and specific. If the answer is not in the context, say:
"I don't have enough data to answer that — try rephrasing or ask about a specific unit or date."

Context (relevant CSV rows):
{context}

Question: {question}

Answer:""")

    # 4. LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=api_key
    )

    # 5. Assemble chain using pure LCEL — no langchain.chains imports at all
    rag_chain = (
        RunnableParallel({
            "context":  retriever | format_docs,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


rag_chain = build_rag_chain(openai_api_key)

if rag_chain is None:
    st.error("⚠️ `final_anomaly_results.csv` not found. Please make sure it is committed to your GitHub repo.")
    st.stop()

# ══════════════════════════════════════════════════════════════
#  CHAT INTERFACE
# ══════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I've indexed the `final_anomaly_results.csv` data into a vector database. "
                "Ask me anything about the solar units, anomalies, yield trends, or risk flags."
            )
        }
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("e.g., Which unit had the sharpest drop in power?"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching the data..."):
            try:
                # LCEL chain returns a plain string directly
                response = rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                error_msg = str(e)
                if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                    st.error("❌ Invalid OpenAI API key. Please check the key entered in the sidebar.")
                elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                    st.error("❌ OpenAI quota exceeded. Check your billing at platform.openai.com.")
                else:
                    st.error(f"❌ Unexpected error: {error_msg}")
