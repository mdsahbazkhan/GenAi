from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel,Field
from typing import List
from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
import streamlit as st
import time

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7,streaming=True)


class BlogState(BaseModel):
    user_input:str
    chat_history:List=Field(default_factory=list)
    topic:str=""
    research:str=""
    draft_blog:str=""
    final_blog:str=""
    
    

def research_agent(state:BlogState) -> BlogState:
    prompt=f"""
    You are a expert blog reasearch agent.
    Your task is to research the given user input and provide a detailed report.
    Topic: {state.user_input}
    """
    res=llm.invoke(prompt)
    return {
        "research":res.content,
        "topic":state.user_input
    }
    
    
def write_agent(state:BlogState) -> BlogState:
    prompt=f"""
    You are a expert blog writer agent.
    Your task is to write a blog based on the research report.
    Research Report: {state.research}
    Topic: {state.topic}
    Requirements:
- Add proper headings
- Add introduction
- Add conclusion
- Use markdown formatting
- Make blog detailed and readable
    """
    res=llm.invoke(prompt)
    return {"draft_blog":res.content}

def editor_agent(state:BlogState) -> BlogState:
    prompt=f"""
    You are a expert blog editor agent.
    Your task is to edit the blog and make it perfect.
    Draft Blog: {state.draft_blog}
    """
    res=llm.invoke(prompt)
    return {"final_blog":res.content}

def update_blog_agent(current_blog, instruction):

    prompt = f"""
    You are an expert blog editor.

    Current Blog:
    {current_blog}

    User Instruction:
    {instruction}

    Update the blog accordingly.
    """

    res = llm.invoke(prompt)

    return res.content


graph=StateGraph(BlogState)

graph.add_node("research",research_agent)
graph.add_node("write", write_agent)
graph.add_node("edit", editor_agent)


graph.add_edge(START,"research")
graph.add_edge("research", "write")
graph.add_edge("write", "edit")
graph.add_edge("edit", END)

app=graph.compile()
st.image(
    app.get_graph().draw_mermaid_png()
)

st.title("AI Blog Generator")
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_blog" not in st.session_state:
    st.session_state.current_blog = ""

for message in st.session_state.messages:
    role=message["role"]
    content=message["content"]
    st.chat_message(role).markdown(content)

user_input=st.chat_input("Enter your blog topic")

if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        edit_keywords = [
    "short",
    "rewrite",
    "improve",
    "seo",
    "professional",
    "expand",
    "summarize"
]

    is_edit = any(word in user_input.lower()for word in edit_keywords)
    
    if is_edit and st.session_state.current_blog:

        final_blog = update_blog_agent(
        st.session_state.current_blog,
        user_input
    )

    else:

        result = app.invoke({
        "user_input": user_input
    })

        final_blog = result["final_blog"]
        # save current blog
    st.session_state.current_blog = final_blog
    st.session_state.messages.append({
    "role": "assistant",
    "content": final_blog
})
    with st.chat_message("assistant"):

        placeholder = st.empty()

        streamed_text = ""

        for chunk in final_blog:

            streamed_text += chunk 

            placeholder.markdown(streamed_text)

            time.sleep(0.001)
