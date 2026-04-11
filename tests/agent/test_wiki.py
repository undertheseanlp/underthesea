"""Tests for WikiAgent - LLM-powered wiki knowledge base manager."""

import json
import os
import shutil
import tempfile
import unittest.mock
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

MOCK_POST = "underthesea.agent.providers._http.post_json"


def _resp(content="Done."):
    return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}


class TestWikiInit(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_init_creates_structure(self):
        from underthesea.agent.wiki import WikiAgent
        w = WikiAgent(self.tmpdir)
        result = w.init(name="Test Wiki", description="A test wiki")

        self.assertTrue((Path(self.tmpdir) / "raw").is_dir())
        self.assertTrue((Path(self.tmpdir) / "wiki").is_dir())
        self.assertTrue((Path(self.tmpdir) / "output").is_dir())
        self.assertTrue((Path(self.tmpdir) / ".wiki.json").exists())
        self.assertTrue((Path(self.tmpdir) / "wiki" / "_index.md").exists())

        config = json.loads((Path(self.tmpdir) / ".wiki.json").read_text())
        self.assertEqual(config["name"], "Test Wiki")

    def test_init_idempotent(self):
        from underthesea.agent.wiki import WikiAgent
        w = WikiAgent(self.tmpdir)
        w.init()
        # Write something in wiki
        (Path(self.tmpdir) / "wiki" / "test.md").write_text("# Test")
        w.init()  # Should not overwrite existing content
        self.assertTrue((Path(self.tmpdir) / "wiki" / "test.md").exists())

    def test_init_new_directory(self):
        from underthesea.agent.wiki import WikiAgent
        new_dir = os.path.join(self.tmpdir, "subdir", "wiki")
        w = WikiAgent(new_dir)
        w.init()
        self.assertTrue(Path(new_dir).is_dir())


class TestWikiTools(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from underthesea.agent.wiki import WikiAgent
        w = WikiAgent(self.tmpdir)
        w.init(name="Test Wiki")

        # Add some test content
        wiki_dir = Path(self.tmpdir) / "wiki"
        (wiki_dir / "concepts").mkdir()
        (wiki_dir / "concepts" / "llm.md").write_text(
            "# Large Language Models\n\nLLMs are neural networks trained on text.\n\n"
            "## Related\n- [transformer](concepts/transformer.md)\n"
        )
        (wiki_dir / "concepts" / "transformer.md").write_text(
            "# Transformer Architecture\n\nThe transformer uses attention mechanisms.\n\n"
            "## Related\n- [llm](concepts/llm.md)\n"
        )

        raw_dir = Path(self.tmpdir) / "raw"
        (raw_dir / "paper.md").write_text(
            "# Attention Is All You Need\n\nThis paper introduces the transformer architecture."
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_wiki_tools_created(self):
        from underthesea.agent.wiki import _make_wiki_tools
        tools = _make_wiki_tools(self.tmpdir)
        tool_names = {t.name for t in tools}
        self.assertIn("wiki_read", tool_names)
        self.assertIn("wiki_write", tool_names)
        self.assertIn("wiki_list", tool_names)
        self.assertIn("wiki_search", tool_names)
        self.assertIn("wiki_delete", tool_names)
        self.assertIn("wiki_stats", tool_names)
        self.assertIn("wiki_backlinks", tool_names)
        self.assertIn("wiki_read_index", tool_names)

    def test_wiki_read(self):
        from underthesea.agent.wiki import _make_wiki_tools
        tools = {t.name: t for t in _make_wiki_tools(self.tmpdir)}
        result = tools["wiki_read"](path="wiki/concepts/llm.md")
        self.assertIn("Large Language Models", result["content"])

    def test_wiki_read_not_found(self):
        from underthesea.agent.wiki import _make_wiki_tools
        tools = {t.name: t for t in _make_wiki_tools(self.tmpdir)}
        result = tools["wiki_read"](path="wiki/nonexistent.md")
        self.assertIn("error", result)

    def test_wiki_write(self):
        from underthesea.agent.wiki import _make_wiki_tools
        tools = {t.name: t for t in _make_wiki_tools(self.tmpdir)}
        result = tools["wiki_write"](path="wiki/topics/new.md", content="# New Topic\n\nContent here.")
        self.assertTrue(result["success"])
        self.assertTrue((Path(self.tmpdir) / "wiki" / "topics" / "new.md").exists())

    def test_wiki_list(self):
        from underthesea.agent.wiki import _make_wiki_tools
        tools = {t.name: t for t in _make_wiki_tools(self.tmpdir)}
        result = tools["wiki_list"](directory="wiki")
        self.assertIn("concepts/", result["dirs"])
        self.assertTrue(any(f["name"] == "_index.md" for f in result["files"]))

    def test_wiki_search(self):
        from underthesea.agent.wiki import _make_wiki_tools
        tools = {t.name: t for t in _make_wiki_tools(self.tmpdir)}
        result = tools["wiki_search"](query="transformer attention")
        self.assertGreater(result["total"], 0)
        paths = [r["path"] for r in result["results"]]
        self.assertIn("wiki/concepts/transformer.md", paths)

    def test_wiki_search_no_results(self):
        from underthesea.agent.wiki import _make_wiki_tools
        tools = {t.name: t for t in _make_wiki_tools(self.tmpdir)}
        result = tools["wiki_search"](query="xyznonexistent")
        self.assertEqual(result["total"], 0)

    def test_wiki_delete(self):
        from underthesea.agent.wiki import _make_wiki_tools
        tools = {t.name: t for t in _make_wiki_tools(self.tmpdir)}
        # Write then delete
        tools["wiki_write"](path="output/temp.md", content="temp")
        result = tools["wiki_delete"](path="output/temp.md")
        self.assertTrue(result["deleted"])
        self.assertFalse((Path(self.tmpdir) / "output" / "temp.md").exists())

    def test_wiki_delete_raw_blocked(self):
        from underthesea.agent.wiki import _make_wiki_tools
        tools = {t.name: t for t in _make_wiki_tools(self.tmpdir)}
        result = tools["wiki_delete"](path="raw/paper.md")
        self.assertIn("error", result)

    def test_wiki_stats(self):
        from underthesea.agent.wiki import _make_wiki_tools
        tools = {t.name: t for t in _make_wiki_tools(self.tmpdir)}
        result = tools["wiki_stats"]()
        self.assertEqual(result["raw_docs"], 1)
        self.assertEqual(result["wiki_articles"], 3)  # _index.md + 2 concepts
        self.assertGreater(result["total_words"], 0)
        self.assertIn("concepts", result["categories"])

    def test_wiki_backlinks(self):
        from underthesea.agent.wiki import _make_wiki_tools
        tools = {t.name: t for t in _make_wiki_tools(self.tmpdir)}
        result = tools["wiki_backlinks"](article="llm")
        self.assertGreater(len(result["backlinks"]), 0)

    def test_wiki_read_index(self):
        from underthesea.agent.wiki import _make_wiki_tools
        tools = {t.name: t for t in _make_wiki_tools(self.tmpdir)}
        result = tools["wiki_read_index"]()
        self.assertTrue(result["exists"])
        self.assertIn("Test Wiki", result["content"])


class TestWikiStats(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from underthesea.agent.wiki import WikiAgent
        self.wiki = WikiAgent(self.tmpdir)
        self.wiki.init(name="Stats Wiki")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_stats(self):
        s = self.wiki.stats()
        self.assertEqual(s["name"], "Stats Wiki")
        self.assertEqual(s["wiki_articles"], 1)  # _index.md
        self.assertEqual(s["raw_docs"], 0)

    def test_stats_with_content(self):
        (Path(self.tmpdir) / "raw" / "doc.md").write_text("raw content")
        (Path(self.tmpdir) / "wiki" / "article.md").write_text("wiki content here")
        s = self.wiki.stats()
        self.assertEqual(s["raw_docs"], 1)
        self.assertEqual(s["wiki_articles"], 2)


class TestWikiSearch(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from underthesea.agent.wiki import WikiAgent
        self.wiki = WikiAgent(self.tmpdir)
        self.wiki.init()
        (Path(self.tmpdir) / "wiki" / "test.md").write_text(
            "# Machine Learning\n\nML is a subset of AI."
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_search(self):
        results = self.wiki.search("machine learning")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Machine Learning")
        self.assertGreater(results[0]["score"], 0)

    def test_search_no_match(self):
        results = self.wiki.search("quantum computing")
        self.assertEqual(len(results), 0)


class TestWikiAgentLLM(TestCase):
    """Tests that require mocking the LLM."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from underthesea.agent.wiki import WikiAgent
        self.wiki = WikiAgent(self.tmpdir)
        self.wiki.init(name="LLM Wiki")
        (Path(self.tmpdir) / "raw" / "doc.md").write_text("# Test Doc\n\nSome content.")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("Compiled 1 document into 2 articles."))
    def test_compile(self, mock_post):
        result = self.wiki.compile()
        self.assertIn("Compiled", result)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("The main theme is testing."))
    def test_ask(self, mock_post):
        result = self.wiki.ask("What is the main theme?")
        self.assertIn("theme", result)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("Lint complete. 0 issues found."))
    def test_lint(self, mock_post):
        result = self.wiki.lint()
        self.assertIn("Lint", result)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("Custom result."))
    def test_custom(self, mock_post):
        result = self.wiki.custom("You are a summary agent.", "Summarize everything.")
        self.assertEqual(result, "Custom result.")

    def test_operations_without_init_fail(self):
        from underthesea.agent.wiki import WikiAgent
        uninit = WikiAgent(os.path.join(self.tmpdir, "nope"))
        with self.assertRaises(FileNotFoundError):
            uninit.compile()
        with self.assertRaises(FileNotFoundError):
            uninit.ask("question")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("Ingested paper.md"))
    def test_ingest_file(self, mock_post):
        # Create a source file outside wiki
        src = os.path.join(self.tmpdir, "external.md")
        Path(src).write_text("# External Article\n\nContent from outside.")
        result = self.wiki.ingest(src)
        self.assertIn("Ingested", result)
        # Check file was copied to raw/
        self.assertTrue((Path(self.tmpdir) / "raw" / "external.md").exists())

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("Ingested web content."))
    def test_ingest_text(self, mock_post):
        result = self.wiki.ingest_text("# Web Article\n\nClipped content.", "web-article.md")
        self.assertIn("Ingested", result)
        self.assertTrue((Path(self.tmpdir) / "raw" / "web-article.md").exists())

    def test_ingest_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.wiki.ingest("/nonexistent/file.md")


class TestUrlDetection(TestCase):
    def test_tweet(self):
        from underthesea.agent.wiki import WikiAgent
        self.assertEqual(WikiAgent._detect_url_type("https://x.com/karpathy/status/123"), "tweet")
        self.assertEqual(WikiAgent._detect_url_type("https://twitter.com/user/status/1"), "tweet")

    def test_arxiv(self):
        from underthesea.agent.wiki import WikiAgent
        self.assertEqual(WikiAgent._detect_url_type("https://arxiv.org/abs/2301.00001"), "arxiv")

    def test_github(self):
        from underthesea.agent.wiki import WikiAgent
        self.assertEqual(WikiAgent._detect_url_type("https://github.com/user/repo"), "github")

    def test_youtube(self):
        from underthesea.agent.wiki import WikiAgent
        self.assertEqual(WikiAgent._detect_url_type("https://youtube.com/watch?v=abc"), "youtube")
        self.assertEqual(WikiAgent._detect_url_type("https://youtu.be/abc"), "youtube")

    def test_webpage(self):
        from underthesea.agent.wiki import WikiAgent
        self.assertEqual(WikiAgent._detect_url_type("https://example.com/article"), "webpage")

    def test_pdf(self):
        from underthesea.agent.wiki import WikiAgent
        self.assertEqual(WikiAgent._detect_url_type("https://example.com/paper.pdf"), "pdf")

    def test_image(self):
        from underthesea.agent.wiki import WikiAgent
        self.assertEqual(WikiAgent._detect_url_type("https://example.com/photo.jpg"), "image")


class TestSafeFilename(TestCase):
    def test_basic(self):
        from underthesea.agent.wiki import WikiAgent
        name = WikiAgent._safe_filename("https://example.com/path/to/page")
        self.assertTrue(name.endswith(".md"))
        self.assertNotIn("://", name)

    def test_long_url_truncated(self):
        from underthesea.agent.wiki import WikiAgent
        name = WikiAgent._safe_filename("https://example.com/" + "a" * 200)
        self.assertLessEqual(len(name), 84)  # 80 + ".md"


class TestNoOverwrite(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_first_write(self):
        from underthesea.agent.wiki import WikiAgent
        dest = Path(self.tmpdir) / "test.md"
        result = WikiAgent._save_no_overwrite(dest, "content")
        self.assertEqual(result, dest)

    def test_no_overwrite(self):
        from underthesea.agent.wiki import WikiAgent
        dest = Path(self.tmpdir) / "test.md"
        dest.write_text("original")
        result = WikiAgent._save_no_overwrite(dest, "new content")
        self.assertNotEqual(result, dest)
        self.assertTrue(result.name.startswith("test_"))
        self.assertEqual(dest.read_text(), "original")  # original untouched


class TestAddNoCompile(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from underthesea.agent.wiki import WikiAgent
        self.wiki = WikiAgent(self.tmpdir)
        self.wiki.init()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_add_file_no_compile(self):
        src = Path(self.tmpdir) / "external.md"
        src.write_text("# Test\n\nContent.")
        result = self.wiki.add(str(src), compile=False)
        self.assertIn("raw", result)
        self.assertTrue((Path(self.tmpdir) / "raw" / "external.md").exists())

    def test_add_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.wiki.add("/nonexistent/file.md")


class TestQAFeedbackLoop(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from underthesea.agent.wiki import WikiAgent
        self.wiki = WikiAgent(self.tmpdir)
        self.wiki.init()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("The answer is 42."))
    def test_ask_saves_qa(self, mock_post):
        self.wiki.ask("What is the answer?")
        output_files = list((Path(self.tmpdir) / "output").glob("qa_*.md"))
        self.assertEqual(len(output_files), 1)
        content = output_files[0].read_text()
        self.assertIn("type: qa", content)
        self.assertIn("What is the answer?", content)
        self.assertIn("The answer is 42.", content)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("Answer."))
    def test_ask_no_save(self, mock_post):
        self.wiki.ask("Question?", save=False)
        output_files = list((Path(self.tmpdir) / "output").glob("qa_*.md"))
        self.assertEqual(len(output_files), 0)


class TestYAMLFrontmatter(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from underthesea.agent.wiki import WikiAgent
        self.wiki = WikiAgent(self.tmpdir)
        self.wiki.init()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("urllib.request.urlopen")
    def test_webpage_frontmatter(self, mock_urlopen):
        mock_resp = unittest.mock.MagicMock()
        mock_resp.read.return_value = b"<html><title>Test Page</title><body><p>Hello</p></body></html>"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = lambda s, *a: None
        mock_urlopen.return_value = mock_resp

        content, _ = self.wiki._fetch_webpage("https://example.com/test")
        self.assertTrue(content.startswith("---\n"))
        self.assertIn("source_url: https://example.com/test", content)
        self.assertIn("type: webpage", content)
        self.assertIn("captured_at:", content)
        self.assertIn("title: \"Test Page\"", content)


class TestWikiImport(TestCase):
    def test_import_from_agent(self):
        from underthesea.agent import WikiAgent
        self.assertIsNotNone(WikiAgent)

    def test_import_direct(self):
        from underthesea.agent.wiki import WikiAgent
        self.assertIsNotNone(WikiAgent)
