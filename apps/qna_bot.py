from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm=ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
import streamlit as st

st.title("AskBuddy-Q&A Bot")
st.markdown("Ask me anything! Type your question below and I'll do my best to answer it.")
if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    role=message["role"]
    content=message["content"]
    st.chat_message(role).markdown(content)

query=st.chat_input("Ask me anything!")

if query:
    st.session_state.messages.append({"role":"user","content":query})
    st.chat_message("user").markdown(query)
    res=llm.invoke(query)
    st.chat_message("ai").markdown(res.text)
    st.session_state.messages.append({"role":"ai","content":res.text})
