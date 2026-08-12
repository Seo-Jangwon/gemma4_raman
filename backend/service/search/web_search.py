"""
[역할] 외부 웹 검색 도구.
  에이전트 내부 지식(KB)에 없는 정보를 물어볼 때 외부에서 찾아온다. 하드웨어와
  무관한 순수 네트워크 조회이며, 결과는 (제목·URL·요약) 목록으로만 돌려준다.

  제공자 선택(자동):
    · GOOGLE_API_KEY + GOOGLE_CSE_ID 환경변수가 있으면 → Google Custom Search API.
      (구글은 무료 공식 검색 API가 없어 키가 필요하다. 키를 넣으면 이 경로를 쓴다.)
    · 없으면 → DuckDuckGo lite(키 불필요). 장비 PC가 인터넷만 되면 바로 동작.

  air-gap(인터넷 불가) 환경이면 네트워크 예외를 잡아 {ok: False, error}로 답하며,
  이 경우 에이전트는 로컬 KB(search_kb)로 폴백하면 된다.
"""
from __future__ import annotations

import html
import os
import re

import httpx

from backend.tools.result import fail, ok

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_TIMEOUT = 10.0


def _clean(fragment: str) -> str:
    """HTML 태그 제거 + 엔티티 언이스케이프 + 공백 정리."""
    text = re.sub(r"<.*?>", "", fragment)
    return html.unescape(text).strip()


def _search_google(query: str, max_results: int) -> dict:
    key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("GOOGLE_CSE_ID")
    r = httpx.get(
        "https://www.googleapis.com/customsearch/v1",
        params={"key": key, "cx": cx, "q": query, "num": min(max_results, 10)},
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    items = r.json().get("items", []) or []
    results = [{"title": it.get("title", ""), "url": it.get("link", ""),
                "snippet": it.get("snippet", "")} for it in items[:max_results]]
    return ok(provider="google", query=query, results=results)


def _search_duckduckgo(query: str, max_results: int) -> dict:
    r = httpx.post("https://lite.duckduckgo.com/lite/", data={"q": query},
                   headers=_UA, timeout=_TIMEOUT, follow_redirects=True)
    r.raise_for_status()
    t = r.text
    # lite 마크업: <a ... href="URL" class='result-link'>TITLE</a>, 요약은 result-snippet td.
    links = re.findall(
        r"""<a[^>]*href=["']([^"']+)["'][^>]*class=["']result-link["'][^>]*>(.*?)</a>""",
        t, re.S)
    snippets = re.findall(r"""class=["']result-snippet["'][^>]*>(.*?)</td>""", t, re.S)
    results = []
    for i, (url, title) in enumerate(links[:max_results]):
        results.append({
            "title": _clean(title),
            "url": url,
            "snippet": _clean(snippets[i]) if i < len(snippets) else "",
        })
    return ok(provider="duckduckgo", query=query, results=results)


def web_search(query: str, max_results: int = 5) -> dict:
    """웹을 검색해 상위 결과(제목·URL·요약)를 돌려준다.

    Returns
    -------
    dict — {ok, provider, query, results:[{title,url,snippet}]} 또는 {ok:False, error}
    """
    query = (query or "").strip()
    if not query:
        return fail("The search query is empty.")
    max_results = max(1, min(int(max_results or 5), 10))

    use_google = bool(os.environ.get("GOOGLE_API_KEY") and os.environ.get("GOOGLE_CSE_ID"))
    try:
        result = _search_google(query, max_results) if use_google \
            else _search_duckduckgo(query, max_results)
    except httpx.HTTPError as e:
        return fail(f"Web search failed (network): {type(e).__name__}. "
                    "Check that the instrument PC is connected to the internet. "
                    "If offline, fall back to search_kb (local knowledge).")
    except Exception as e:
        return fail(f"Web search failed: {e}")

    if not result.get("results"):
        result["note"] = ("No web results. Rephrase the query, or try search_kb "
                          "for the local knowledge base.")
    return result
