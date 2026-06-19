"""
Simple Chat example using MCPAgent with built-in conversation memory.
This example demonstrates how to use the MCPAgent class to create a simple chat application.

Special thanks to https://github.com/microsoft/playwright-mcp for the inspiration.
"""
import asyncio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient

async def run_memory_chat():

    """Run Chat using MCPAgent with built-in conversation memory."""
    # Load environment variables
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    config_file = "browser_mcp.json"
    print("Initializing Chat....")

    # Create MCPClient from config file
    client = MCPClient.from_config_file(config_file)

    # Create LLM
    llm = ChatGroq(model="llama-3.1-8b-instant")


    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=15,memory_enabled=True)

    print("Chat initialized. Starting conversation...")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("Type 'clear' to clear the conversation memory.")
    print("===================")

    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            elif user_input.lower() == "clear":
                agent.clear_conversation_memory()
                print("Conversation memory cleared.")
                continue
            try:
                response = await agent.run(user_input, max_steps=10)
                print(f"\nResponse: {response}")
                print("===================")
            except Exception as e:
                print(f"Error: {e}")
                print("===================")
                continue
    finally:
        if client and client.sessions:
            await client.close_all_sessions()
        print("Chat session ended.")


if __name__ == "__main__":
    asyncio.run(run_memory_chat())