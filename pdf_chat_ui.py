import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import FakeEmbeddings
from langchain_openai  import OpenAIEmbeddings

embeddings = FakeEmbeddings(size=1536)
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os

load_dotenv()

st.set_page_config(page_title="PDF Chat Agent", page_icon="📄", layout="wide")
st.title("📄 PDF Chat Agent")
st.caption("Upload a PDF and ask questions about it")

# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
#def get_embedding_model():
    #return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#def get_embedding_model():
    #return FakeEmbeddings(size=1536)
def get_embedding_model():
    return OpenAIEmbeddings()
if "vector_store" not in st.session_state:
    st.warning("Please upload a PDF first")
    st.stop()
@st.cache_resource(show_spinner="Processing PDF...")
def build_vector_store(_docs):
    embedding = get_embedding_model()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=9)
    chunks = splitter.split_documents(_docs)
    return FAISS.from_documents(chunks, embedding)

def format_docs(retrieved_docs):
    return "\n\n".join([doc.page_content for doc in retrieved_docs])

def build_chain(retriever):
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    prompt = PromptTemplate(
        template="""You are a helpful assistant.
Answer the question based on the provided context only.
If the context is insufficient, say you don't know.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    return parallel_chain | prompt | llm | StrOutputParser()

# ── Sidebar — PDF Upload ──────────────────────────────────────────────────────

with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Indexing PDF..."):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            st.session_state.vector_store = build_vector_store(tuple(docs))
            st.session_state.chat_history = []
        os.unlink(tmp_path)
        st.success(f"✅ Indexed {len(docs)} pages")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ── Main — Chat Interface ─────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask something about your PDF..."):
    if "vector_store" not in st.session_state:
        st.warning("⬅️ Please upload a PDF first.")
    else:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity", search_kwargs={"k": 5}
                )
                chain = build_chain(retriever)
                answer = chain.invoke(question)
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})