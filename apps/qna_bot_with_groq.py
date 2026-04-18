## LLM
## Tool Google Search Tool
## Agent
##Memory
##Streaming
## Web Interface


from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="QuickAnswer AI",
    page_icon="🤖",
    layout="centered"
)
llm=ChatGroq(model="llama-3.3-70b-versatile",streaming=True,temperature=0.7)
search = GoogleSerperAPIWrapper()
tools=[search.run]
memory=MemorySaver()
if "memory" not in st.session_state:
    st.session_state.memory=memory
    st.session_state.history=[]
agent=create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a helpful AI assistant.

Rules:
- Give short and clear answers
- Use bullet points when possible
- Avoid long paragraphs
""",
    checkpointer=st.session_state.memory
)
## Building a simple web interface using streamlit

st.markdown("""
<h2 style='text-align: center;'>🤖 QuickAnswer AI</h2>
<p style='text-align: center; color: gray;'>
Real-time AI Agent with Memory + Search
</p>
""", unsafe_allow_html=True)

for message in st.session_state.history:
    role=message["role"]
    content=message["content"]
    st.chat_message(role).markdown(content)
query=st.chat_input("Ask me anything: ")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user","content":query})
     
    response=agent.stream({"messages":[{"role":"user","content":query}]},{"configurable":{"thread_id":"qna_bot_thread"}},stream_mode="messages")
    # answer=response["messages"][-1].content
    # st.chat_message("ai").markdown(answer)
    ai_container=st.chat_message("ai")
    with ai_container:
        space=st.empty()
        message=""
        
        for chunk in response:
            message=message + chunk[0].content
            space.write(message)
        st.session_state.history.append({"role":"ai","content":message.replace(". ", ".\n\n")})