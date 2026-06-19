from mcp.server.fastmcp import FastMCP

mcp=FastMCP("weather")

@mcp.tool()
async def get_weather(location:str) -> str:
    """   Get the current weather for any city.
    Always use this tool when the user asks about weather."""

    return f"The weather in {location} is rainy."

if __name__=="__main__":
    mcp.run(transport="streamable-http")