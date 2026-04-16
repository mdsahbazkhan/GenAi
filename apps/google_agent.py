from dotenv import load_dotenv
load_dotenv()
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)
search = GoogleSerperAPIWrapper()
memory=MemorySaver()
agent=create_agent(
    model=model,
    tools=[search.run],
    system_prompt="You are a agent and can search for any questions on google.",
    checkpointer=memory
)

while True:
    query=input("User: ")
    if query.lower() in ["exit", "quit"]:
        print("Exiting the agent. Goodbye!")
        break
    
    response=agent.invoke({"messages":[{"role":"user","content":query}]},{"configurable":{"thread_id":"google_agent_thread"}})   
    print("Ai: ", response["messages"][-1].content)