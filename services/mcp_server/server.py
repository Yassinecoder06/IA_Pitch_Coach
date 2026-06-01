import asyncio
import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: List[Dict[str, str]] = []
        self._capture_title = False
        self._current_title_parts: List[str] = []
        self._current_url = ""

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        if tag != "a":
            return
        attrs_dict = dict(attrs)
        class_attr = attrs_dict.get("class", "")
        if "result__a" in class_attr:
            self._capture_title = True
            self._current_title_parts = []
            self._current_url = attrs_dict.get("href", "")

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self._current_title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture_title:
            title = "".join(self._current_title_parts).strip()
            if title and self._current_url:
                self.results.append({"title": title, "url": self._current_url})
            self._capture_title = False
            self._current_title_parts = []
            self._current_url = ""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._chunks.append(text)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


def _fetch_url(url: str, timeout: int = 10) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (MCP Web Fetcher)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    html = raw.decode("utf-8", errors="ignore")
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


def _search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    params = urllib.parse.urlencode({"q": query})
    url = f"https://duckduckgo.com/html/?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (MCP Web Search)"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read()
    html = raw.decode("utf-8", errors="ignore")
    parser = _DuckDuckGoParser()
    parser.feed(html)
    return parser.results[:max_results]


server = Server("ia-pitch-coach-tools")


@server.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="web_search",
            description="Search the public web via DuckDuckGo (no API key required).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="fetch_url",
            description="Fetch a web page and return readable text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute URL"},
                },
                "required": ["url"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    if name == "web_search":
        query = (arguments.get("query") or "").strip()
        max_results = int(arguments.get("max_results") or 5)
        if not query:
            return [TextContent(type="text", text="Missing query.")]
        results = _search_duckduckgo(query, max_results=max_results)
        if not results:
            return [TextContent(type="text", text="No results found.")]
        lines = []
        for idx, result in enumerate(results, start=1):
            lines.append(f"{idx}. {result['title']}\n{result['url']}")
        return [TextContent(type="text", text="\n\n".join(lines))]

    if name == "fetch_url":
        url = (arguments.get("url") or "").strip()
        if not url:
            return [TextContent(type="text", text="Missing url.")]
        try:
            text = _fetch_url(url)
            return [TextContent(type="text", text=text or "No text extracted.")]
        except Exception as exc:
            return [TextContent(type="text", text=f"Fetch failed: {exc}")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
