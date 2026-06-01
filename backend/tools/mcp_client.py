import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
except Exception:  # pragma: no cover
    ClientSession = None  # type: ignore
    StdioServerParameters = None  # type: ignore
    stdio_client = None  # type: ignore


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SERVER_PATH = str(ROOT_DIR / "services" / "mcp_server" / "server.py")


def _is_enabled() -> bool:
    return os.getenv("MCP_ENABLED", "false").lower() == "true"


def _auto_search_enabled() -> bool:
    return os.getenv("MCP_AUTO_SEARCH", "false").lower() == "true"


def _max_results() -> int:
    try:
        return max(1, min(10, int(os.getenv("MCP_MAX_RESULTS", "5"))))
    except ValueError:
        return 5


def tool_awareness_prompt() -> str:
    if not _is_enabled() or ClientSession is None or StdioServerParameters is None or stdio_client is None:
        return ""

    auto_search = "enabled" if _auto_search_enabled() else "disabled"
    return (
        "External tool runtime: MCP is enabled for this app. "
        "Available tools are web_search for public DuckDuckGo web results and fetch_url "
        "for readable page text. Automatic web search is currently "
        f"{auto_search}. When web snippets are provided, use them as source context; "
        "when they are missing but current facts matter, say what should be searched or verified."
    )


def _build_command() -> tuple[str, list[str]]:
    cmd = os.getenv("MCP_SERVER_CMD", "python").strip() or "python"
    args_raw = os.getenv("MCP_SERVER_ARGS", "").strip()
    if args_raw:
        args = shlex.split(args_raw)
    else:
        args = [DEFAULT_SERVER_PATH]
    return cmd, args


def _extract_text(result: Any) -> str:
    if result is None:
        return ""
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    if not content:
        return ""

    chunks = []
    for item in content:
        text = getattr(item, "text", None)
        if text is None and isinstance(item, dict):
            text = item.get("text")
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


async def call_tool(tool_name: str, args: Dict[str, Any]) -> str:
    if not _is_enabled() or ClientSession is None or StdioServerParameters is None or stdio_client is None:
        return ""

    cmd, cmd_args = _build_command()
    server_params = StdioServerParameters(
        command=cmd,
        args=cmd_args,
        cwd=str(ROOT_DIR),
    )

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, args)
                return _extract_text(result)
    except Exception:
        return ""


async def maybe_web_search(transcript: str) -> str:
    if not _auto_search_enabled():
        return ""

    query = _infer_query(transcript)
    if not query:
        return ""

    return await call_tool("web_search", {
        "query": query,
        "max_results": _max_results(),
    })


def _infer_query(text: str) -> str:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return ""

    explicit_match = re.search(r"(?:research|search|lookup)\s*:\s*(.+)$", cleaned, re.IGNORECASE)
    if explicit_match:
        return explicit_match.group(1).strip()[:180]

    keywords = r"\b(market|competitor|pricing|regulation|regulatory|news|trend|benchmark)\b"
    if re.search(keywords, cleaned, re.IGNORECASE):
        return cleaned[:180]

    return ""
