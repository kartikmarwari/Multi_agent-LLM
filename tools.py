from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """Search the web and return structured results"""
    
    results = tavily.search(
        query=query,
        max_results=8,
        search_depth="advanced"
    )

    out = []
    for r in results["results"]:
        out.append(
    f"""
{{
  "title": "{r['title']}",
  "url": "{r['url']}",
  "summary": "{r['content'][:200]}"
}}
"""
        )

    return "\n====================\n".join(out)


@tool
def scrape_url(url: str) -> str:
    """Scrape clean content from URL"""

    try:
        resp = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        clean_text = soup.get_text(separator=" ", strip=True)

        if len(clean_text) < 500:
            return ""

        return clean_text[:3000]

    except Exception as e:
        return f"Scraping failed: {str(e)}"


# =========================
# NEW TOOL: multi_search
# (used by search_agent when it gets split sub-questions,
# saves it from calling web_search one by one manually)
# =========================
@tool
def multi_search(queries: str) -> str:
    """Search multiple queries and return CLEAN markdown"""

    query_list = [q.strip() for q in queries.split("|") if q.strip()]
    final_md = []

    for q in query_list:
        results = tavily.search(
            query=q,
            max_results=5,
            search_depth="advanced"
        )

        section = [f"## 🔎 {q}\n"]

        for r in results["results"]:
            section.append(f"""
### {r['title']}
- 🔗 {r['url']}
- 📝 {r['content'][:200]}...
""")

        final_md.append("\n".join(section))

    return "\n\n---\n\n".join(final_md)