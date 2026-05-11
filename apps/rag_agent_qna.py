from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st
import os
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
}

h1 {
    color: white;
    text-align: center;
    font-size: 3rem;
}

.upload-box {
    border: 2px dashed #4F8BF9;
    padding: 25px;
    border-radius: 15px;
    background-color: #161A23;
    text-align: center;
    margin-bottom: 20px;
}

.stChatMessage {
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

### data in st session state

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
    
if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
    
if "messages" not in st.session_state:
    st.session_state.messages = []

def process_document(path):
    
    ## Load the PDF document
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    ## Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents=docs)

    ## Create embeddings for the document chunks
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    vector_db=InMemoryVectorStore.from_documents(
        documents=texts,
        embedding=embeddings
    )

    ## Create an agent -tool,llm,prompt
    llm=ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    @tool
    def retrieve_context(query:str):
        """Retreive documents relevent to a query from the knowledge base."""
        contexts=""
        
        docs = vector_db.similarity_search(
            query=f"""
            Find relevant information for:
            {query}

            Focus on course details, duration, videos,
            projects, modules, and prerequisites.
            """,
            k=4
        )
        for doc in docs:
            contexts += doc.page_content + "\n\n"
            
        return contexts

    system_prompt = """
    You are a helpful assistant that answers questions using retrieved context.

    My knowledge base consists of details from uploaded documents.

    Always use the retrieve_context tool before answering any question related to the documents.
    """

    memory=InMemorySaver()

    agent=create_agent(
        model=llm, 
        tools=[retrieve_context], 
        system_prompt=system_prompt,
        checkpointer=memory
        )
    
    st.session_state.agent=agent
    st.session_state.document_uploaded=True

st.title("🤖 AI PDF RAG Assistant")

st.markdown(
    """
    <div class="upload-box">
        <h3>Upload your PDF documents and chat with AI</h3>
        <p>Powered by LangChain + Groq + Gemini Embeddings</p>
    </div>
    """,
    unsafe_allow_html=True
)

### Upload UI
if not st.session_state.document_uploaded:

    st.subheader("📄 Upload Documents")

    uploaded = st.file_uploader(
        label="Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:

        with st.spinner("⚡ Processing documents..."):

            path = "./doc_files/"

            os.makedirs(path, exist_ok=True)

            for file in uploaded:

                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())

            process_document(path)

            st.success("✅ Documents processed successfully!")

            st.rerun()

### Chat UI

if st.session_state.document_uploaded and st.session_state.agent:

    st.subheader("💬 Chat With Your Documents")

    for message in st.session_state.messages:

        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            st.markdown(content)

    query = st.chat_input("Ask anything from your PDF...")

    if query:

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("🤖 Thinking..."):

            response = st.session_state.agent.invoke(
                {
                    "messages": [
                        {"role": "user", "content": query}
                    ]
                },
                {"configurable": {"thread_id": "1"}}
            )

            answer = response["messages"][-1].content

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })