from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


async def make_nws_request(url:str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling"""
    headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json",
    }
    async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers,timeout=30.0)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                print(f"Error making request to NWS API: {e}")
                return None
    

def format_alert(feature:dict) -> str:
    """Format a weather alert feature into a human-readable string"""
    props=feature["properties"]

    return f"""
    Event: {props.get("event", "Unknown")}
    Severity: {props.get("severity", "Unknown")}
    Description: {props.get("description", "Unknown")}
    Instructions: {props.get("instruction", "Unknown")}
    """

@mcp.tool()
async def get_alerts(state:str) ->str:
    """Set weather alerts for a USA State.
    Args:
        state: The two-letter state code (e.g. "CA", "NY")
    """
    url=f"{NWS_API_BASE}/alerts/active/area/{state}"
    data=await make_nws_request(url)

    if not data or "features" not in data:
        return "unable to fetch alerts or no alerts found"

    if not data["features"]:
        return "no active alerts found"
    alerts=[format_alert(feature) for feature in data["features"]]
    return "\n\n".join(alerts)