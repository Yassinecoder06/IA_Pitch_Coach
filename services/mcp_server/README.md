# MCP Tool Server (Standalone)

This is a standalone MCP server that exposes web tools for the AI Pitch Coach runtime.
It runs over stdio and provides two tools:
- `web_search` (DuckDuckGo HTML search, no API key)
- `fetch_url` (download a page and return readable text)

## Setup

```bash
cd services/mcp_server
pip install -r requirements.txt
```

## Run

```bash
python server.py
```

## Connect from your LLM runtime

Use your MCP client to launch this server with:

```json
{
  "command": "python",
  "args": ["services/mcp_server/server.py"],
  "env": {}
}
```

The client will discover tools via MCP `list_tools` and can call them using `call_tool`.
