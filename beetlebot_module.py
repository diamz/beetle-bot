
#!/usr/bin/env python3
"""
Async website crawler that collects site structure, pages, clean main text, links,
plus images (with alt tags), schema.org JSON-LD, and schema keywords.

New:
- Boilerplate removal (headers/footers/menus/cookie banners)
- Main-content extraction with link-density heuristic or Trafilatura
- Switch via --extractor [heuristic|full|trafilatura]
"""
from __future__ import annotations

import asyncio
import csv
import json
import re
import sys
import time
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple, Set
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode, urldefrag

import httpx
from bs4 import BeautifulSoup
import tldextract
import urllib.robotparser as robotparser

# Optional high-quality extractor
try:
    import trafilatura  # pip install trafilatura
    HAS_TRAF = True
except Exception:
    HAS_TRAF = False

DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 WebCrawler/1.2"
)
TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}
HTML_MIME_PREFIXES = ("text/html", "application/xhtml+xml")
ASSET_EXTS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
              ".csv", ".zip", ".rar", ".7z", ".mp3", ".mp4", ".jpg", ".png"}

@dataclass
class PageRecord:
    url: str
    parent_url: Optional[str]
    depth: int
    status: Optional[int]
    content_type: Optional[str]
    title: Optional[str]
    meta_description: Optional[str]
    headings: Dict[str, List[str]]
    text: Optional[str]
    links: Dict[str, List[str]]
    canonical: Optional[str]
    last_modified: Optional[str]
    discovered_at: str
    images: List[Dict[str, Optional[str]]]
    schemas: List[dict]
    schema_keywords: List[str]

# -----------------------------
# URL + robots helpers
# -----------------------------

def normalize_url(raw_url: str) -> str:
    url, _ = urldefrag(raw_url)
    p = urlparse(url)
    scheme = p.scheme.lower() if p.scheme else "http"
    netloc = p.hostname.lower() if p.hostname else ""
    # keep non-default port
    if p.port and not ((scheme == "http" and p.port == 80) or (scheme == "https" and p.port == 443)):
        netloc = f"{netloc}:{p.port}"
    path = re.sub(r"/+", "/", p.path or "/")
    query_pairs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k not in TRACKING_PARAMS]
    query_pairs.sort()
    query = urlencode(query_pairs)
    return urlunparse((scheme, netloc, path, "", query, ""))

def is_same_domain(url: str, allowed_domains: Set[str], include_subdomains: bool) -> bool:
    host = urlparse(url).hostname or ""
    ext = tldextract.extract(host)
    domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    if include_subdomains:
        return any(domain == ad or host.endswith("." + ad) for ad in allowed_domains)
    else:
        return any(domain == ad or host == ad for ad in allowed_domains)

def robots_for(start_url: str, user_agent: str) -> robotparser.RobotFileParser:
    p = urlparse(start_url)
    robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        pass
    return rp

# -----------------------------
# Content extraction (clean main body)
# -----------------------------

def _visible_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True))


def _remove_boilerplate_nodes(soup: BeautifulSoup) -> None:
    # Drop common layout/boilerplate blocks
    for tag in soup.find_all(["header", "footer", "nav", "aside", "form"]):
        tag.decompose()
    # ARIA roles
    for tag in soup.select('[role="navigation"], [role="banner"], [role="contentinfo"], [aria-label*="breadcrumb" i]'):
        tag.decompose()
    # Heuristic by id/class names
    patterns = re.compile(r"(nav|menu|header|footer|breadcrumb|subscribe|cookie|consent|popup|modal|advert|share|social)", re.I)
    for el in soup.find_all(True, {"id": patterns}):
        el.decompose()
    for el in soup.find_all(True, {"class": patterns}):
        el.decompose()


def _link_density(node: BeautifulSoup) -> float:
    text = node.get_text(separator=" ", strip=True) if node else ""
    if not text:
        return 1.0
    link_text = " ".join(a.get_text(separator=" ", strip=True) for a in node.find_all("a"))
    return min(1.0, (len(link_text) + 1) / (len(text) + 1))


def _heuristic_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    _remove_boilerplate_nodes(soup)
    candidates = soup.find_all(["article", "main", "section", "div"])
    best_node, best_score = None, -1.0
    for n in candidates:
        txt = n.get_text(separator=" ", strip=True)
        if len(txt) < 200:
            continue
        ld = _link_density(n)
        score = len(txt) * (1.0 - ld)
        if score > best_score:
            best_score, best_node = score, n
    if best_node is None:
        return _visible_text(soup)
    _remove_boilerplate_nodes(best_node)
    return re.sub(r"\s+", " ", best_node.get_text(separator=" ", strip=True))


def extract_main_text(html: str, mode: str = "heuristic") -> str:
    """Return clean main content. Modes: 'heuristic' (default), 'full', 'trafilatura'."""
    if mode == "full":
        return _visible_text(BeautifulSoup(html, "lxml"))
    if mode == "trafilatura" and HAS_TRAF:
        try:
            out = trafilatura.extract(html, include_comments=False, include_tables=False)
            if out:
                return re.sub(r"\s+", " ", out.strip())
        except Exception:
            pass
    return _heuristic_main_text(html)

# -----------------------------
# Other extractors
# -----------------------------

def extract_links(base_url: str, soup: BeautifulSoup) -> List[str]:
    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href").strip()
        if href.startswith(("javascript:", "mailto:", "tel:")):
            continue
        abs_url = normalize_url(urljoin(base_url, href))
        if abs_url.startswith("http"):
            links.append(abs_url)
    return links


def extract_meta(soup: BeautifulSoup, base_url: str):
    title = soup.title.string.strip() if soup.title and soup.title.string else None

    md = None
    m = soup.find("meta", attrs={"name": "description"})
    if m and m.get("content"):
        md = m["content"].strip()
        md = m["content"].strip()

    headings = {
        "h1": [h.get_text(strip=True) for h in soup.find_all("h1")],
        "h2": [h.get_text(strip=True) for h in soup.find_all("h2")],
        "h3": [h.get_text(strip=True) for h in soup.find_all("h3")],
    }

    canonical = None
    link_tag = soup.find("link", rel=lambda v: v and "canonical" in v)
    if link_tag and link_tag.get("href"):
        canonical = normalize_url(urljoin(base_url, link_tag["href"]))

    return title, md, headings, canonical


def extract_images(base_url: str, soup: BeautifulSoup) -> List[Dict[str, Optional[str]]]:
    imgs = []
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        abs_src = normalize_url(urljoin(base_url, src))
        imgs.append({
            "src": abs_src,
            "alt": img.get("alt"),
            "title": img.get("title")
        })
    return imgs


def extract_schemas(soup: BeautifulSoup) -> List[dict]:
    results = []
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
        except Exception:
            continue
    return results


def extract_schema_keywords(schemas: List[dict]) -> List[str]:
    kws: List[str] = []
    for s in schemas:
        if not isinstance(s, dict):
            continue
        if "keywords" in s:
            val = s["keywords"]
            if isinstance(val, str):
                kws.extend([k.strip() for k in val.split(",")])
            elif isinstance(val, list):
                kws.extend([str(k).strip() for k in val])
        if "@graph" in s and isinstance(s["@graph"], list):
            kws.extend(extract_schema_keywords(s["@graph"]))
    return sorted(set(kws))

# -----------------------------
# Crawler
# -----------------------------
class Crawler:
    def __init__(self, start_url: str, max_pages: int = 500, max_depth: int = 5, concurrency: int = 10,
                delay: float = 0.3, user_agent: str = DEFAULT_UA, respect_robots: bool = True,
                out_dir: Optional[str] = "./out", extractor: str = "heuristic"):
        self.start_url = normalize_url(start_url)
        self.allowed_domains = {self._registered_domain(urlparse(self.start_url).hostname or "")}
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.concurrency = concurrency
        self.delay = delay
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        self.out_dir = (out_dir.rstrip("/") if isinstance(out_dir, str) else None)
        self.extractor = extractor
        self.visited: Set[str] = set()
        self.edges: List[Tuple[str, str]] = []
        self.assets: List[Tuple[str, str]] = []
        self.sem = asyncio.Semaphore(concurrency)
        self.rp = robots_for(self.start_url, self.user_agent) if self.respect_robots else None
        self.client = httpx.AsyncClient(headers={"User-Agent": self.user_agent}, http2=True, follow_redirects=True)

    @staticmethod
    def _registered_domain(host: str) -> str:
        ext = tldextract.extract(host)
        return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

    def _allowed(self, url: str) -> bool:
        if not is_same_domain(url, self.allowed_domains, True):
            return False
        if self.respect_robots and self.rp is not None:
            try:
                return self.rp.can_fetch(self.user_agent, url)
            except Exception:
                return True
        return True


    async def crawl(self):
        q: asyncio.Queue[Tuple[str, Optional[str], int]] = asyncio.Queue()
        await q.put((self.start_url, None, 0))

        # hard cap handled atomically
        pages_used = 0
        budget_lock = asyncio.Lock()
        stop_enqueuing = False

        await self._prepare_outputs()

        async def worker():
            nonlocal pages_used, stop_enqueuing
            while True:
                try:
                    url, parent, depth = await asyncio.wait_for(q.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    break  # queue empty long enough

                try:
                    # Drain mode: once cap hit, do not process, just drop tasks fast
                    if stop_enqueuing:
                        continue

                    # quick filters first
                    if url in self.visited:
                        continue
                    self.visited.add(url)

                    if depth > self.max_depth:
                        continue
                    if not self._allowed(url):
                        continue

                    # ====== ATOMIC BUDGET RESERVATION ======
                    allocated = False
                    async with budget_lock:
                        if pages_used < self.max_pages:
                            pages_used += 1
                            allocated = True
                            # if we just consumed the last slot, stop any further enqueues
                            if pages_used >= self.max_pages:
                                stop_enqueuing = True
                    if not allocated:
                        # no budget left → drain
                        continue
                    # =======================================

                    # fetch & parse with timeout and concurrency guard
                    async with self.sem:
                        try:
                            rec, discovered = await asyncio.wait_for(
                                self._fetch_and_parse(url, parent, depth),
                                timeout=20.0
                            )
                        except asyncio.TimeoutError:
                            print(f"[TIMEOUT] {url}")
                            rec, discovered = None, []

                    if rec:
                        print(f"[OK] {url} status={rec.status}")
                        await self._write_page(rec)

                        # Enqueue only if still allowed
                        if not stop_enqueuing:
                            for link in discovered:
                                if link not in self.visited:
                                    await q.put((link, url, depth + 1))
                    else:
                        print(f"[FAIL] {url}")

                except Exception as e:
                    print(f"[ERROR] Worker crashed on {url}: {e}")

                finally:
                    q.task_done()
                    if self.delay:
                        await asyncio.sleep(self.delay)

        workers = [asyncio.create_task(worker()) for _ in range(self.concurrency)]
        try:
            await q.join()  # waits until every queued task got a task_done()
        finally:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            await self.client.aclose()
            await self._flush_edges_assets()




    async def _fetch_and_parse(self, url: str, parent: Optional[str], depth: int):
        try:
            r = await self.client.get(url, timeout=20)
        except Exception as e:
            print(f"[ERROR] {url} ({e})"); return None, []
        status = r.status_code
        ct = r.headers.get("content-type", "").split(";")[0].strip().lower()
        last_mod = r.headers.get("last-modified")
        discovered_at = datetime.now(timezone.utc).isoformat()

        if status != 200:
            print(f"[WARN] {url} returned {status}")

        # Non-HTML: record as asset
        if not any(ct.startswith(pfx) for pfx in HTML_MIME_PREFIXES):
            self.assets.append((url, ct or "binary"))
            rec = PageRecord(url, parent, depth, status, ct, None, None,
                             {"h1": [], "h2": [], "h3": []}, None,
                             {"internal": [], "external": []}, None,
                             last_mod, discovered_at,
                             images=[], schemas=[], schema_keywords=[])
            return rec, []

        html = r.text or ""
        soup = BeautifulSoup(html, "lxml")

        title, md, headings, canonical = extract_meta(soup, url)
        # *** Clean main text ***
        text = extract_main_text(html, mode=self.extractor)

        raw_links = extract_links(url, soup)
        internal_links, external_links = [], []
        for lk in raw_links:
            if is_same_domain(lk, self.allowed_domains, True):
                internal_links.append(lk)
                if parent and lk:
                    self.edges.append((parent, lk))
            else:
                external_links.append(lk)

        schemas = extract_schemas(soup)
        images = extract_images(url, soup)
        schema_keywords = extract_schema_keywords(schemas)

        rec = PageRecord(url, parent, depth, status, ct, title, md, headings,
                         text[:20000] if text else None,
                         {"internal": internal_links, "external": external_links},
                         canonical, last_mod, discovered_at,
                         images=images, schemas=schemas, schema_keywords=schema_keywords)

        discovered_next = [lk for lk in ([canonical] if canonical else []) + internal_links if lk and self._allowed(lk)]
        return rec, discovered_next

    async def _prepare_outputs(self):
        if self.out_dir is None:
            return
        import os
        os.makedirs(self.out_dir, exist_ok=True)
        open(f"{self.out_dir}/pages.jsonl", "w", encoding="utf-8").close()
        with open(f"{self.out_dir}/edges.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["source_url", "target_url"])
        with open(f"{self.out_dir}/assets.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["url", "type_or_ext"])

    async def _write_page(self, rec: PageRecord):
        if self.out_dir is None:
            return  # no file writes
        with open(f"{self.out_dir}/pages.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    async def _flush_edges_assets(self):
        if self.out_dir is None:
            return
        with open(f"{self.out_dir}/edges.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(self.edges)
        with open(f"{self.out_dir}/assets.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(self.assets)

# -----------------------------
# CLI
# -----------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Async site crawler with clean main-content extraction")
    ap.add_argument("start_url")
    ap.add_argument("--max-pages", type=int, default=50)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--user-agent", default=DEFAULT_UA)
    ap.add_argument("--no-robots", action="store_true")
    ap.add_argument("--out", default="./out")
    ap.add_argument("--extractor", choices=["heuristic", "full", "trafilatura"], default="heuristic",
                    help="Text extraction strategy for page body (default: heuristic)")
    return ap.parse_args(argv)

def main(argv=None):
    ns = parse_args(argv)
    crawler = Crawler(
        start_url=ns.start_url,
        max_pages=ns.max_pages,
        max_depth=ns.max_depth,
        concurrency=ns.concurrency,
        delay=ns.delay,
        user_agent=ns.user_agent,
        respect_robots=not ns.no_robots,
        out_dir=ns.out,
        extractor=ns.extractor,
    )
    t0 = time.time()
    asyncio.run(crawler.crawl())
    print(f"Done in {time.time()-t0:.1f}s. Pages: {len(crawler.visited)} | Edges: {len(crawler.edges)} | Assets: {len(crawler.assets)}")
    print(f"Outputs in: {crawler.out_dir}")

if __name__ == "__main__":
    sys.exit(main())

# ------------------------------------------------------------------------------------------------
import asyncio
from typing import List, Dict, Optional
from dataclasses import asdict
import concurrent.futures

async def _run_crawler_async(start_url: str,
                             max_pages: int = 500,
                             max_depth: int = 5,
                             concurrency: int = 10,
                             delay: float = 0.3,
                             user_agent: str = DEFAULT_UA,
                             respect_robots: bool = True,
                             extractor: str = "heuristic",
                             out_dir: Optional[str] = None,
                             progress_callback=None) -> List[Dict]:
    """
    Run the crawler asynchronously and return a list of JSON-like dicts.
    If out_dir is None, no files are written.
    """
    crawler = Crawler(
        start_url=start_url,
        max_pages=max_pages,
        max_depth=max_depth,
        concurrency=concurrency,
        delay=delay,
        user_agent=user_agent,
        respect_robots=respect_robots,
        out_dir=out_dir,  # None = no file writes
        extractor=extractor,
    )

    results: List[Dict] = []

    async def _collect_page(rec: PageRecord):
        results.append(asdict(rec))
        if progress_callback:
            progress_callback(len(results), asdict(rec))

    # capture into memory instead of files
    crawler._write_page = _collect_page  # type: ignore[attr-defined]

    await crawler.crawl()
    return results

def _run_in_thread(func, *args, **kwargs):
    """Run an async entrypoint in a dedicated thread with its own event loop, return the result."""
    def runner():
        return asyncio.run(func(*args, **kwargs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(runner)
        return fut.result()


def run_crawler(start_url: str,
                max_pages: int = 500,
                max_depth: int = 5,
                concurrency: int = 10,
                delay: float = 0.3,
                user_agent: str = DEFAULT_UA,
                respect_robots: bool = True,
                extractor: str = "heuristic",
                out_dir: Optional[str] = None,
                progress_callback=None) -> List[Dict]:
    """
    Blocking wrapper that ALWAYS returns a list of JSONL-style dicts.
    - In a normal script: uses asyncio.run() directly.
    - Inside an already-running event loop (Streamlit/Jupyter/etc.): runs in a separate thread.
    """
    try:
        loop = asyncio.get_running_loop()
        # If there IS a running loop, offload to a background thread
        if loop.is_running():
            return _run_in_thread(_run_crawler_async, start_url, max_pages, max_depth,
                                  concurrency, delay, user_agent, respect_robots, extractor, out_dir,progress_callback)
    except RuntimeError:
        # No running loop
        pass

    # Plain Python execution
    return asyncio.run(_run_crawler_async(start_url, max_pages, max_depth,
                                          concurrency, delay, user_agent, respect_robots, extractor, out_dir,progress_callback))
