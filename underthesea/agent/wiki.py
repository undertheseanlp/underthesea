"""WikiAgent - LLM-powered personal knowledge base manager.

Inspired by Karpathy's approach: raw data is collected into raw/,
compiled by an LLM into a .md wiki, then operated on for Q&A,
linting, and incremental enhancement. The LLM maintains the wiki,
you rarely touch it directly.

Wiki directory structure:
    my-wiki/
    ├── raw/              # Raw source documents
    ├── wiki/             # Compiled wiki articles (LLM-maintained)
    │   └── _index.md     # Master index
    ├── output/           # Generated outputs (slides, charts)
    └── .wiki.json        # Configuration

Usage:
    from underthesea.agent import WikiAgent

    wiki = WikiAgent("./my-wiki")
    wiki.init()
    wiki.ingest("article.md")
    wiki.compile()
    answer = wiki.ask("What is the main theme?")
    wiki.lint()
"""

import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from underthesea.agent.agent import Agent
from underthesea.agent.tools import Tool

# ============================================================================
# Wiki directory management
# ============================================================================

DEFAULT_CONFIG = {
    "name": "My Wiki",
    "description": "",
    "version": "1.0",
    "max_iterations": 15,
}


def _ensure_wiki(wiki_dir: str) -> Path:
    """Validate that a wiki directory exists and is initialized."""
    p = Path(wiki_dir)
    if not (p / ".wiki.json").exists():
        raise FileNotFoundError(
            f"Not a wiki directory: {wiki_dir}\n"
            f"Run WikiAgent('{wiki_dir}').init() first."
        )
    return p


# ============================================================================
# Wiki Tools - functions the agent can call
# ============================================================================


def _make_wiki_tools(wiki_dir: str) -> list[Tool]:
    """Create wiki-specific tools bound to a wiki directory."""
    root = Path(wiki_dir).resolve()

    # -- read article / raw doc --
    def wiki_read(path: str) -> dict:
        """Read a file from the wiki. Path is relative to wiki root.
        Examples: 'wiki/concepts/llm.md', 'raw/paper.md', 'wiki/_index.md'
        """
        full = root / path
        if not full.exists():
            return {"error": f"File not found: {path}"}
        try:
            content = full.read_text(encoding="utf-8")
            return {"path": path, "content": content, "size": len(content)}
        except Exception as e:
            return {"path": path, "error": str(e)}

    # -- write article --
    def wiki_write(path: str, content: str) -> dict:
        """Write or update a wiki article. Path is relative to wiki root.
        Use paths under 'wiki/' for compiled articles, 'output/' for generated content.
        Creates parent directories automatically.
        """
        full = root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        try:
            full.write_text(content, encoding="utf-8")
            return {"path": path, "success": True, "bytes": len(content)}
        except Exception as e:
            return {"path": path, "error": str(e)}

    # -- list files --
    def wiki_list(directory: str = "") -> dict:
        """List files in a wiki directory. Defaults to wiki root.
        Common directories: 'raw', 'wiki', 'wiki/concepts', 'output'
        Returns markdown files and subdirectories.
        """
        target = root / directory
        if not target.exists():
            return {"directory": directory, "error": "Directory not found"}

        files = []
        dirs = []
        for item in sorted(target.iterdir()):
            if item.name.startswith("."):
                continue
            if item.is_dir():
                dirs.append(item.name + "/")
            elif item.suffix in (".md", ".txt", ".html", ".json", ".csv"):
                stat = item.stat()
                files.append({
                    "name": item.name,
                    "size": stat.st_size,
                    "modified": int(stat.st_mtime),
                })
        return {"directory": directory or ".", "dirs": dirs, "files": files}

    # -- search wiki --
    def wiki_search(query: str, directory: str = "wiki") -> dict:
        """Full-text search across wiki articles. Returns matching files with
        context snippets. Searches in the given directory (default: 'wiki').
        """
        target = root / directory
        if not target.exists():
            return {"query": query, "results": [], "error": "Directory not found"}

        query_lower = query.lower()
        query_terms = query_lower.split()
        results = []

        for md_file in sorted(target.rglob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8")
            except Exception:
                continue

            text_lower = text.lower()
            # Score: count how many query terms appear
            score = sum(1 for t in query_terms if t in text_lower)
            if score == 0:
                continue

            # Extract snippet around first match
            rel_path = str(md_file.relative_to(root))
            snippet = ""
            for term in query_terms:
                idx = text_lower.find(term)
                if idx >= 0:
                    start = max(0, idx - 80)
                    end = min(len(text), idx + len(term) + 80)
                    snippet = "..." + text[start:end].replace("\n", " ") + "..."
                    break

            results.append({
                "path": rel_path,
                "score": score,
                "snippet": snippet,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return {"query": query, "total": len(results), "results": results[:20]}

    # -- delete file --
    def wiki_delete(path: str) -> dict:
        """Delete a wiki article or file. Path is relative to wiki root.
        Cannot delete raw/ source documents (use for wiki/ and output/ only).
        """
        if path.startswith("raw/") or path.startswith("raw\\"):
            return {"error": "Cannot delete raw source documents"}
        full = root / path
        if not full.exists():
            return {"error": f"File not found: {path}"}
        try:
            full.unlink()
            return {"path": path, "deleted": True}
        except Exception as e:
            return {"path": path, "error": str(e)}

    # -- wiki stats --
    def wiki_stats() -> dict:
        """Get statistics about the wiki: article count, word count,
        raw doc count, categories, etc.
        """
        stats = {"raw_docs": 0, "wiki_articles": 0, "output_files": 0,
                 "total_words": 0, "categories": []}

        raw_dir = root / "raw"
        wiki_dir_path = root / "wiki"
        output_dir = root / "output"

        if raw_dir.exists():
            stats["raw_docs"] = sum(1 for _ in raw_dir.rglob("*.md"))

        if wiki_dir_path.exists():
            word_count = 0
            categories = set()
            for md in wiki_dir_path.rglob("*.md"):
                stats["wiki_articles"] += 1
                try:
                    text = md.read_text(encoding="utf-8")
                    word_count += len(text.split())
                except Exception:
                    pass
                # Track subdirectories as categories
                rel = md.relative_to(wiki_dir_path)
                if len(rel.parts) > 1:
                    categories.add(rel.parts[0])
            stats["total_words"] = word_count
            stats["categories"] = sorted(categories)

        if output_dir.exists():
            stats["output_files"] = sum(1 for _ in output_dir.rglob("*") if _.is_file())

        return stats

    # -- get backlinks --
    def wiki_backlinks(article: str) -> dict:
        """Find all wiki articles that link to the given article.
        Article is the filename without extension (e.g. 'transformer').
        """
        wiki_path = root / "wiki"
        if not wiki_path.exists():
            return {"article": article, "backlinks": []}

        pattern = re.compile(
            r"\[.*?\]\(.*?" + re.escape(article) + r".*?\)", re.IGNORECASE
        )
        backlinks = []
        for md in wiki_path.rglob("*.md"):
            try:
                text = md.read_text(encoding="utf-8")
                if pattern.search(text):
                    backlinks.append(str(md.relative_to(root)))
            except Exception:
                continue

        return {"article": article, "backlinks": backlinks}

    # -- read index --
    def wiki_read_index() -> dict:
        """Read the wiki master index (_index.md). This is the main
        entry point to understand the wiki structure.
        """
        idx = root / "wiki" / "_index.md"
        if not idx.exists():
            return {"exists": False, "content": ""}
        content = idx.read_text(encoding="utf-8")
        return {"exists": True, "content": content}

    # Build Tool objects
    return [
        Tool(wiki_read, name="wiki_read",
             description="Read a file from the wiki. Path is relative to wiki root (e.g. 'wiki/concepts/llm.md', 'raw/paper.md')."),
        Tool(wiki_write, name="wiki_write",
             description="Write or update a wiki article. Path relative to wiki root. Use 'wiki/' for articles, 'output/' for generated content."),
        Tool(wiki_list, name="wiki_list",
             description="List files and subdirectories. Pass directory relative to wiki root (e.g. 'raw', 'wiki', 'wiki/concepts'). Empty string for root."),
        Tool(wiki_search, name="wiki_search",
             description="Full-text search across wiki articles. Returns matching files with context snippets."),
        Tool(wiki_delete, name="wiki_delete",
             description="Delete a wiki or output file. Cannot delete raw source documents."),
        Tool(wiki_stats, name="wiki_stats",
             description="Get wiki statistics: article count, word count, categories, raw doc count."),
        Tool(wiki_backlinks, name="wiki_backlinks",
             description="Find all articles that link to the given article name."),
        Tool(wiki_read_index, name="wiki_read_index",
             description="Read the wiki master index (_index.md) to understand wiki structure."),
    ]


# ============================================================================
# System prompts for different modes
# ============================================================================

COMPILE_PROMPT = """\
You are a wiki compiler agent. Your job is to process raw source documents
and compile them into a well-structured wiki.

For each raw document:
1. Read it carefully and extract key concepts, facts, and relationships.
2. Create or update wiki articles under wiki/ with clear, well-written summaries.
3. Organize articles into categories using subdirectories (e.g. wiki/concepts/, wiki/topics/).
4. Add backlinks between related articles using markdown links.
5. Update wiki/_index.md with a master index of all articles and brief summaries.

Guidelines:
- Each article should have a clear title (# heading), summary, and content sections.
- Use [[article-name]] style links for cross-references within the wiki.
- Add a "## Sources" section linking back to raw/ documents.
- Add a "## Related" section with links to related wiki articles.
- Keep articles focused on one concept/topic each.
- Use wiki/_index.md as the master table of contents.
"""

QA_PROMPT = """\
You are a wiki Q&A agent. You have access to a knowledge base wiki.

When answering questions:
1. First read the wiki index to understand available content.
2. Search the wiki for relevant articles.
3. Read the most relevant articles in full.
4. Synthesize a comprehensive answer based on the wiki content.
5. Cite your sources by referencing specific wiki articles.

If the wiki doesn't contain enough information, say so clearly.
If the answer requires information not in the wiki, note what's missing.
"""

LINT_PROMPT = """\
You are a wiki quality assurance agent. Your job is to audit the wiki
and improve its overall data integrity.

Perform these checks:
1. **Broken links**: Find articles that reference non-existent articles.
2. **Orphan articles**: Find articles with no backlinks from other articles.
3. **Missing summaries**: Articles without proper summary sections.
4. **Inconsistencies**: Contradictory information across articles.
5. **Missing index entries**: Articles not listed in _index.md.
6. **Stale content**: Raw docs that haven't been compiled into wiki articles.
7. **Suggested connections**: Concepts that should be linked but aren't.

For each issue found:
- Describe the problem clearly.
- Fix it if possible (update the article, add links, update index).
- If not fixable automatically, note it as a recommendation.

Write a lint report to output/lint-report.md with all findings and actions taken.
"""

INGEST_PROMPT = """\
You are a wiki ingestion agent. A new raw document has been added.

Your task:
1. Read the new raw document.
2. Read the current wiki index to understand existing content.
3. Extract key concepts from the new document.
4. For each concept: create a new article or update an existing one.
5. Add proper cross-references and backlinks.
6. Update wiki/_index.md with any new articles.

The new document is: {raw_path}
"""


# ============================================================================
# WikiAgent class
# ============================================================================


class WikiAgent:
    """LLM-powered wiki knowledge base manager.

    Manages a wiki directory with raw sources, compiled articles, and outputs.
    Uses the underthesea Agent framework with wiki-specific tools.

    Parameters
    ----------
    wiki_dir : str or Path
        Path to the wiki directory.
    provider : BaseProvider or LLM, optional
        LLM provider. Auto-detects from env vars if not specified.
    max_iterations : int
        Max tool-calling iterations per agent call.

    Examples
    --------
    >>> wiki = WikiAgent("./my-wiki")
    >>> wiki.init()
    >>> wiki.ingest("path/to/article.md")
    >>> wiki.compile()
    >>> answer = wiki.ask("What are the key concepts?")
    >>> wiki.lint()
    """

    def __init__(self, wiki_dir: str | Path, provider=None, max_iterations: int = 15):
        self.wiki_dir = Path(wiki_dir).resolve()
        self._provider = provider
        self._max_iterations = max_iterations

    def init(self, name: str = "My Wiki", description: str = "") -> Path:
        """Initialize a new wiki directory structure.

        Creates raw/, wiki/, output/ directories and .wiki.json config.

        Returns
        -------
        Path
            The wiki directory path.
        """
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        (self.wiki_dir / "raw").mkdir(exist_ok=True)
        (self.wiki_dir / "wiki").mkdir(exist_ok=True)
        (self.wiki_dir / "output").mkdir(exist_ok=True)

        config_path = self.wiki_dir / ".wiki.json"
        if not config_path.exists():
            config = {**DEFAULT_CONFIG, "name": name, "description": description}
            config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))

        index_path = self.wiki_dir / "wiki" / "_index.md"
        if not index_path.exists():
            index_path.write_text(
                f"# {name}\n\n{description}\n\n"
                "## Articles\n\n_No articles yet. Run compile to process raw documents._\n"
            )

        return self.wiki_dir

    def init_from_existing(self, name: str | None = None, description: str = "") -> Path:
        """Initialize a wiki from a directory that already has .md files.

        Moves existing .md/.txt/.html files into raw/, then sets up
        the standard wiki structure (wiki/, output/, .wiki.json).

        Parameters
        ----------
        name : str, optional
            Wiki name. Defaults to the directory name.
        description : str
            Wiki description.

        Returns
        -------
        Path
            The wiki directory path.
        """
        if name is None:
            name = self.wiki_dir.name.replace("-", " ").replace("_", " ").title()

        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = self.wiki_dir / "raw"
        raw_dir.mkdir(exist_ok=True)

        # Move existing content files into raw/
        moved = []
        for ext in ("*.md", "*.txt", "*.html"):
            for f in sorted(self.wiki_dir.glob(ext)):
                if f.parent == self.wiki_dir:  # Only top-level files
                    dest = raw_dir / f.name
                    f.rename(dest)
                    moved.append(f.name)

        # Now init the rest of the structure
        self.init(name=name, description=description)
        return self.wiki_dir

    def _make_agent(self, instruction: str) -> Agent:
        """Create an Agent with wiki tools and the given instruction."""
        tools = _make_wiki_tools(str(self.wiki_dir))
        return Agent(
            name="wiki-agent",
            tools=tools,
            instruction=instruction,
            max_iterations=self._max_iterations,
            provider=self._provider,
        )

    # -- URL type detection (inspired by graphify) --

    @staticmethod
    def _detect_url_type(url: str) -> str:
        """Classify URL for targeted extraction."""
        lower = url.lower()
        if "twitter.com" in lower or "x.com" in lower:
            return "tweet"
        if "arxiv.org" in lower:
            return "arxiv"
        if "github.com" in lower:
            return "github"
        if "youtube.com" in lower or "youtu.be" in lower:
            return "youtube"
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.lower()
        if path.endswith(".pdf"):
            return "pdf"
        if any(path.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif")):
            return "image"
        return "webpage"

    @staticmethod
    def _safe_filename(url: str, suffix: str = ".md") -> str:
        """Turn a URL into a safe filename."""
        parsed = urllib.parse.urlparse(url)
        name = parsed.netloc + parsed.path
        name = re.sub(r"[^\w\-]", "_", name).strip("_")
        name = re.sub(r"_+", "_", name)[:80]
        return name + suffix

    @staticmethod
    def _save_no_overwrite(dest: Path, content: str) -> Path:
        """Write content to dest, appending counter if file already exists."""
        if not dest.exists():
            dest.write_text(content, encoding="utf-8")
            return dest
        counter = 1
        while True:
            new_dest = dest.with_stem(f"{dest.stem}_{counter}")
            if not new_dest.exists():
                new_dest.write_text(content, encoding="utf-8")
                return new_dest
            counter += 1

    # -- add: main entry point for URLs and files --

    def add(self, source: str, compile: bool = True) -> str:
        """Add a source to the wiki. Accepts a URL or file path.

        Fetches content, saves to raw/ with YAML frontmatter metadata.
        Optionally compiles into wiki articles.

        Parameters
        ----------
        source : str
            URL (http/https) or file path.
        compile : bool
            If True, run LLM to compile into wiki articles after saving.

        Returns
        -------
        str
            Path of saved file (compile=False) or agent's compilation response.
        """
        _ensure_wiki(str(self.wiki_dir))

        if source.startswith("http://") or source.startswith("https://"):
            content, filename = self._fetch_url(source)
        else:
            # Local file
            src = Path(source)
            if not src.exists():
                raise FileNotFoundError(f"Source file not found: {source}")
            content = src.read_text(encoding="utf-8")
            filename = src.name

        # Save to raw/ (no overwrite)
        dest = self._save_no_overwrite(self.wiki_dir / "raw" / filename, content)
        raw_rel = f"raw/{dest.name}"

        if not compile:
            return str(dest)

        # Run ingestion agent
        instruction = INGEST_PROMPT.format(raw_path=raw_rel)
        agent = self._make_agent(instruction)
        return agent(f"Ingest and compile the new document: {raw_rel}")

    # -- URL fetchers --

    def _fetch_url(self, url: str) -> tuple[str, str]:
        """Fetch URL content based on detected type. Returns (content, filename)."""
        url_type = self._detect_url_type(url)

        if url_type == "tweet":
            return self._fetch_tweet(url)
        if url_type == "arxiv":
            return self._fetch_arxiv(url)
        return self._fetch_webpage(url)

    def _fetch_tweet(self, url: str) -> tuple[str, str]:
        """Fetch tweet content via fxtwitter API with YAML frontmatter."""
        parsed = urllib.parse.urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 3 and path_parts[1] == "status":
            username, status_id = path_parts[0], path_parts[2]
        else:
            raise RuntimeError(f"Cannot parse tweet URL: {url}")

        api_url = f"https://api.fxtwitter.com/{username}/status/{status_id}"
        req = urllib.request.Request(
            api_url, headers={"User-Agent": "underthesea-wiki/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to fetch tweet: {e}")

        tweet = data.get("tweet", {})
        author = tweet.get("author", {})
        author_name = author.get("name", username)
        text = tweet.get("text", "")
        created = tweet.get("created_at", "")
        now = datetime.now(timezone.utc).isoformat()

        content = (
            f"---\n"
            f"source_url: {url}\n"
            f"type: tweet\n"
            f"author: {author_name}\n"
            f"author_handle: \"@{username}\"\n"
            f"date: \"{created}\"\n"
            f"captured_at: \"{now}\"\n"
            f"---\n\n"
            f"# {author_name}: {text[:80]}...\n\n"
            f"{text}\n"
        )
        filename = f"{username}-{status_id}.md"
        return content, filename

    def _fetch_arxiv(self, url: str) -> tuple[str, str]:
        """Fetch arXiv paper metadata and abstract with YAML frontmatter."""
        arxiv_id_match = re.search(r"(\d{4}\.\d{4,5})", url)
        if not arxiv_id_match:
            return self._fetch_webpage(url)

        arxiv_id = arxiv_id_match.group(1)
        api_url = f"https://export.arxiv.org/abs/{arxiv_id}"
        req = urllib.request.Request(
            api_url, headers={"User-Agent": "underthesea-wiki/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
        except Exception:
            return self._fetch_webpage(url)

        # Parse metadata
        title_match = re.search(
            r'class="title[^"]*"[^>]*>(.*?)</h1>', html, re.DOTALL | re.IGNORECASE
        )
        title = re.sub(r"<[^>]+>", " ", title_match.group(1)).strip() if title_match else arxiv_id

        abstract_match = re.search(
            r'class="abstract[^"]*"[^>]*>(.*?)</blockquote>', html, re.DOTALL | re.IGNORECASE
        )
        abstract = re.sub(r"<[^>]+>", "", abstract_match.group(1)).strip() if abstract_match else ""

        authors_match = re.search(
            r'class="authors"[^>]*>(.*?)</div>', html, re.DOTALL | re.IGNORECASE
        )
        paper_authors = re.sub(r"<[^>]+>", "", authors_match.group(1)).strip() if authors_match else ""
        now = datetime.now(timezone.utc).isoformat()

        content = (
            f"---\n"
            f"source_url: {url}\n"
            f"type: paper\n"
            f"arxiv_id: \"{arxiv_id}\"\n"
            f"title: \"{title}\"\n"
            f"authors: \"{paper_authors}\"\n"
            f"captured_at: \"{now}\"\n"
            f"---\n\n"
            f"# {title}\n\n"
            f"**Authors:** {paper_authors}\n"
            f"**arXiv:** {arxiv_id}\n\n"
            f"## Abstract\n\n{abstract}\n"
        )
        filename = f"arxiv_{arxiv_id.replace('.', '_')}.md"
        return content, filename

    def _fetch_webpage(self, url: str) -> tuple[str, str]:
        """Fetch and convert a webpage to markdown with YAML frontmatter."""
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; underthesea-wiki/1.0)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}")

        # Extract title
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else "Untitled"

        markdown = self._html_to_markdown(html)
        now = datetime.now(timezone.utc).isoformat()

        content = (
            f"---\n"
            f"source_url: {url}\n"
            f"type: webpage\n"
            f"title: \"{title}\"\n"
            f"captured_at: \"{now}\"\n"
            f"---\n\n"
            f"# {title}\n\n"
            f"{markdown[:12000]}\n"
        )
        filename = self._safe_filename(url)
        return content, filename

    @staticmethod
    def _html_to_markdown(html: str) -> str:
        """Best-effort HTML to markdown conversion using only stdlib."""
        for tag in ("script", "style", "nav", "footer", "header", "aside"):
            html = re.sub(
                rf"<{tag}[\s>].*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE
            )
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
        for i in range(1, 7):
            html = re.sub(
                rf"<h{i}[^>]*>(.*?)</h{i}>",
                lambda m, level=i: f"\n{'#' * level} {m.group(1).strip()}\n",
                html, flags=re.DOTALL | re.IGNORECASE,
            )
        html = re.sub(r"<(?:p|div)[^>]*>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"</(?:p|div)>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"<(?:b|strong)[^>]*>(.*?)</(?:b|strong)>", r"**\1**", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<(?:i|em)[^>]*>(.*?)</(?:i|em)>", r"*\1*", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(
            r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            r"[\2](\1)", html, flags=re.DOTALL | re.IGNORECASE,
        )
        html = re.sub(r"<li[^>]*>", "\n- ", html, flags=re.IGNORECASE)
        html = re.sub(r"</li>", "", html, flags=re.IGNORECASE)
        html = re.sub(r"<[^>]+>", "", html)
        html = html.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        html = html.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
        lines = [line.strip() for line in html.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def ingest(self, source_path: str | Path) -> str:
        """Ingest a source document into raw/ and compile it into the wiki.

        Parameters
        ----------
        source_path : str or Path
            Path to the source file (.md, .txt, .html) to ingest.

        Returns
        -------
        str
            Agent's response describing what was compiled.
        """
        _ensure_wiki(str(self.wiki_dir))
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Copy to raw/
        dest = self.wiki_dir / "raw" / source.name
        dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

        # Run ingestion agent
        raw_rel = f"raw/{source.name}"
        instruction = INGEST_PROMPT.format(raw_path=raw_rel)
        agent = self._make_agent(instruction)
        return agent(f"Ingest and compile the new document: {raw_rel}")

    def ingest_text(self, content: str, filename: str) -> str:
        """Ingest raw text content directly (e.g. from web clipper).

        Parameters
        ----------
        content : str
            The text content to ingest.
        filename : str
            Filename for the raw document (e.g. 'article-about-llms.md').

        Returns
        -------
        str
            Agent's response describing what was compiled.
        """
        _ensure_wiki(str(self.wiki_dir))
        dest = self.wiki_dir / "raw" / filename
        dest.write_text(content, encoding="utf-8")

        raw_rel = f"raw/{filename}"
        instruction = INGEST_PROMPT.format(raw_path=raw_rel)
        agent = self._make_agent(instruction)
        return agent(f"Ingest and compile the new document: {raw_rel}")

    def compile(self) -> str:
        """Compile all raw documents into wiki articles.

        Reads everything in raw/, generates/updates wiki articles,
        maintains the index, and creates cross-references.

        Returns
        -------
        str
            Agent's response describing compilation results.
        """
        _ensure_wiki(str(self.wiki_dir))
        agent = self._make_agent(COMPILE_PROMPT)
        return agent(
            "Compile the wiki: read all raw documents, create/update wiki articles, "
            "organize into categories, add cross-references, and update the index."
        )

    def ask(self, question: str, save: bool = True) -> str:
        """Ask a question against the wiki knowledge base.

        The answer is saved to output/ as a feedback loop, so queries
        accumulate knowledge in the wiki (inspired by graphify).

        Parameters
        ----------
        question : str
            The question to answer.
        save : bool
            If True, save Q&A result to output/ for future reference.

        Returns
        -------
        str
            Agent's answer based on wiki content.
        """
        _ensure_wiki(str(self.wiki_dir))
        agent = self._make_agent(QA_PROMPT)
        answer = agent(question)

        if save:
            self._save_qa(question, answer)

        return answer

    def _save_qa(self, question: str, answer: str) -> Path:
        """Save Q&A result to output/ with YAML frontmatter (feedback loop)."""
        output_dir = self.wiki_dir / "output"
        output_dir.mkdir(exist_ok=True)

        now = datetime.now(timezone.utc)
        slug = re.sub(r"[^\w]", "_", question.lower())[:50].strip("_")
        filename = f"qa_{now.strftime('%Y%m%d_%H%M%S')}_{slug}.md"

        content = (
            f"---\n"
            f"type: qa\n"
            f"date: \"{now.isoformat()}\"\n"
            f"question: \"{question.replace(chr(34), chr(39))}\"\n"
            f"---\n\n"
            f"# Q: {question}\n\n"
            f"## Answer\n\n"
            f"{answer}\n"
        )

        dest = output_dir / filename
        dest.write_text(content, encoding="utf-8")
        return dest

    def lint(self) -> str:
        """Run health checks on the wiki.

        Checks for broken links, orphan articles, inconsistencies,
        missing index entries, and suggests improvements.

        Returns
        -------
        str
            Agent's lint report.
        """
        _ensure_wiki(str(self.wiki_dir))
        agent = self._make_agent(LINT_PROMPT)
        return agent(
            "Run a full health check on the wiki. Check for broken links, orphans, "
            "inconsistencies, missing entries, and suggest improvements. "
            "Write the report to output/lint-report.md"
        )

    def search(self, query: str) -> list[dict]:
        """Search the wiki without using the LLM.

        Simple full-text search for quick lookups.

        Parameters
        ----------
        query : str
            Search query.

        Returns
        -------
        list[dict]
            Matching articles with paths, scores, and snippets.
        """
        _ensure_wiki(str(self.wiki_dir))
        wiki_path = self.wiki_dir / "wiki"
        if not wiki_path.exists():
            return []

        query_lower = query.lower()
        terms = query_lower.split()
        results = []

        for md in sorted(wiki_path.rglob("*.md")):
            try:
                text = md.read_text(encoding="utf-8")
            except Exception:
                continue

            text_lower = text.lower()
            score = sum(1 for t in terms if t in text_lower)
            if score == 0:
                continue

            # Extract title from first heading
            title = md.stem
            for line in text.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            # Snippet
            snippet = ""
            for term in terms:
                idx = text_lower.find(term)
                if idx >= 0:
                    start = max(0, idx - 60)
                    end = min(len(text), idx + len(term) + 60)
                    snippet = text[start:end].replace("\n", " ").strip()
                    break

            results.append({
                "path": str(md.relative_to(self.wiki_dir)),
                "title": title,
                "score": score,
                "snippet": snippet,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def stats(self) -> dict:
        """Get wiki statistics without using the LLM.

        Returns
        -------
        dict
            Statistics: raw_docs, wiki_articles, total_words, categories, etc.
        """
        _ensure_wiki(str(self.wiki_dir))
        result = {
            "name": "",
            "raw_docs": 0,
            "wiki_articles": 0,
            "output_files": 0,
            "total_words": 0,
            "categories": [],
        }

        # Read config
        config_path = self.wiki_dir / ".wiki.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            result["name"] = config.get("name", "")

        # Count raw docs
        raw_dir = self.wiki_dir / "raw"
        if raw_dir.exists():
            result["raw_docs"] = sum(
                1 for f in raw_dir.rglob("*") if f.is_file() and f.suffix in (".md", ".txt", ".html")
            )

        # Count wiki articles
        wiki_path = self.wiki_dir / "wiki"
        if wiki_path.exists():
            categories = set()
            for md in wiki_path.rglob("*.md"):
                result["wiki_articles"] += 1
                try:
                    result["total_words"] += len(md.read_text(encoding="utf-8").split())
                except Exception:
                    pass
                rel = md.relative_to(wiki_path)
                if len(rel.parts) > 1:
                    categories.add(rel.parts[0])
            result["categories"] = sorted(categories)

        # Count output files
        output_dir = self.wiki_dir / "output"
        if output_dir.exists():
            result["output_files"] = sum(1 for f in output_dir.rglob("*") if f.is_file())

        return result

    def custom(self, instruction: str, message: str) -> str:
        """Run the agent with a custom instruction and message.

        Useful for ad-hoc operations like generating slides,
        creating summaries, or any custom wiki operation.

        Parameters
        ----------
        instruction : str
            System instruction for the agent.
        message : str
            User message / task description.

        Returns
        -------
        str
            Agent's response.
        """
        _ensure_wiki(str(self.wiki_dir))
        agent = self._make_agent(instruction)
        return agent(message)
