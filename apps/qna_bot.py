from dotenv import load_dotenv
load_dotenv()

# from langchain_google_genai import ChatGoogleGenerativeAI
# llm=ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
from langchain_groq import ChatGroq
import os
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
     api_key=os.getenv("GROQ_API_KEY"),
     streaming=True
)
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
    
    conversation=""
    for message in st.session_state.messages:
        conversation+=f"{message['role']}: {message['content']}\n"
        # AI response container
    with st.chat_message("ai"):
        response_placeholder = st.empty()
        full_response = ""
    res=llm.stream(conversation)
    for chunk in res:
        full_response+=chunk.text
        response_placeholder.markdown(full_response)

#Save the AI response to session state
    st.session_state.messages.append({"role":"ai","content":full_response})