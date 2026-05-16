import os
import platform
import signal
import subprocess
import sys
from pathlib import Path

import click

from underthesea.corpus.validate_corpus import DEFAULT_MAX_ERROR, validate_corpus
from underthesea.data_fetcher import DataFetcher
from underthesea.datasets.vlsp2013_wtk.revise_corpus import revise_corpus
from underthesea.model_fetcher import ModelFetcher


@click.group()
def main(args=None):
    """Console script for underthesea"""
    pass


@main.command()
@click.option('-a', '--all', is_flag=True, required=False)
def list_model(all):
    ModelFetcher.list(all)


@main.command()
@click.argument('model', required=True)
def download_model(model):
    ModelFetcher.download(model)


@main.command()
@click.argument('text', required=True)
def tts(text):
    from underthesea.pipeline.tts import tts as underthesea_tts
    underthesea_tts(text, play=True)


@main.command()
@click.argument('audio', required=False)
@click.option('-m', '--model', default='openai/whisper-small', help='ASR model name')
@click.option('-l', '--language', default='vi', help='Language code (e.g. vi, en)')
@click.option('-o', '--outfile', default=None,
              help='Save microphone recording to this file (only when AUDIO is omitted)')
def transcribe(audio, model, language, outfile):
    """Auto transcribe voice. Pass an AUDIO file path, or omit to record from the mic."""
    if audio:
        from underthesea.pipeline.transcribe import transcribe as _transcribe
        text = _transcribe(audio, model=model, language=language)
    else:
        from underthesea.pipeline.transcribe import auto_transcribe
        text = auto_transcribe(outfile=outfile, model=model, language=language)
    click.echo(text)


@main.command()
@click.argument('model', required=True)
def remove_model(model):
    ModelFetcher.remove(model)


@main.command()
@click.option('-a', '--all', is_flag=True, required=False)
def list_data(all):
    DataFetcher.list(all)


@main.command()
@click.argument('dataset', required=True)
@click.argument('url', required=False)
@click.option('-o', '--output-folder', required=False, help='Output folder for downloaded data')
def download_data(dataset, url, output_folder):
    DataFetcher.download_data(dataset, url, output_folder)


@main.command()
@click.argument('data', required=True)
def remove_data(data):
    DataFetcher.remove(data)


@main.command()
@click.option('-t', '--type', required=True)
@click.option('-c', '--corpus', required=True)
@click.option('--max-error', default=DEFAULT_MAX_ERROR, type=int)
def validate(type, corpus, max_error):
    validate_corpus(type, corpus, max_error)


@main.command()
@click.option('-c', '--corpus', required=True)
def revise(corpus):
    revise_corpus(corpus)


@main.command()
def info():
    from underthesea.version import __version__ as underthesea_version
    python_version = platform.python_version()
    system_info = f"{platform.system()}{platform.release()}"
    print("")
    print("ENVIRONMENT")
    print(f" underthesea version : {underthesea_version}")
    print(f"      python version : {python_version}")
    print(f"  system information : {system_info}")

    print("")
    print("MODULES")
    print("       sent_tokenize : OK")
    print("      text_normalize : OK")
    print("       word_tokenize : OK")
    print("             pos_tag : OK")
    print("               chunk : OK")
    print("                 ner : OK")
    print("            classify : OK")
    print("           sentiment : PARTIAL")
    print("    dependency_parse : SUSPENDED")
    print("         lang_detect : OK")
    print("           resources : OK")


@main.command()
@click.option('--backend-port', default=8001, help='Backend API port')
@click.option('--frontend-port', default=3000, help='Frontend port')
def chat(backend_port, frontend_port):
    """Start the Underthesea Chat application (frontend + backend)."""
    # Find chat app directory
    underthesea_dir = Path(__file__).resolve().parent.parent
    chat_dir = underthesea_dir / "extensions" / "apps" / "chat"
    backend_dir = chat_dir / "backend"
    frontend_dir = chat_dir / "frontend"

    if not chat_dir.exists():
        click.echo(f"Error: Chat app not found at {chat_dir}")
        click.echo("Please ensure the chat app is installed.")
        sys.exit(1)

    # Check backend node_modules
    if not (backend_dir / "node_modules").exists():
        click.echo("Installing backend dependencies...")
        subprocess.run(["npm", "install"], cwd=backend_dir, check=True)

    # Check frontend node_modules
    if not (frontend_dir / "node_modules").exists():
        click.echo("Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)

    processes = []

    def cleanup(signum=None, frame=None):
        click.echo("\nShutting down...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # Start backend (Node.js)
        click.echo(f"Starting backend on http://localhost:{backend_port}")
        backend_proc = subprocess.Popen(
            ["node", "src/index.js"],
            cwd=backend_dir,
            env={**os.environ, "PORT": str(backend_port)}
        )
        processes.append(backend_proc)

        # Start frontend
        click.echo(f"Starting frontend on http://localhost:{frontend_port}")
        frontend_proc = subprocess.Popen(
            ["npm", "run", "dev", "--", "-p", str(frontend_port)],
            cwd=frontend_dir
        )
        processes.append(frontend_proc)

        click.echo("")
        click.echo("Underthesea Chat is running!")
        click.echo(f"  Frontend: http://localhost:{frontend_port}")
        click.echo(f"  Backend:  http://localhost:{backend_port}")
        click.echo("")
        click.echo("Press Ctrl+C to stop")

        # Wait for processes
        for p in processes:
            p.wait()

    except Exception as e:
        click.echo(f"Error: {e}")
        cleanup()


_WIKI_CONFIG_DIR = Path.home() / ".config" / "underthesea"
_WIKI_CONFIG_FILE = _WIKI_CONFIG_DIR / "wiki.json"


def _get_wiki_dir(wiki_dir):
    """Resolve wiki directory: -w flag > config (use) > cwd."""
    if wiki_dir is not None:
        return wiki_dir
    # Read from config
    if _WIKI_CONFIG_FILE.exists():
        import json as _json
        cfg = _json.loads(_WIKI_CONFIG_FILE.read_text())
        return cfg.get("current_wiki", ".")
    return "."


@main.group()
def wiki():
    """LLM-powered wiki knowledge base manager."""
    pass


@wiki.command()
@click.argument('path')
def use(path):
    """Set the current wiki directory.

    \b
    Example:
      underthesea wiki use /path/to/my-wiki
      underthesea wiki stats  # now uses that wiki
    """
    import json as _json
    resolved = str(Path(path).resolve())
    if not Path(resolved, ".wiki.json").exists():
        click.echo(f"Error: {resolved} is not a wiki (no .wiki.json found)")
        sys.exit(1)
    _WIKI_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {}
    if _WIKI_CONFIG_FILE.exists():
        cfg = _json.loads(_WIKI_CONFIG_FILE.read_text())
    cfg["current_wiki"] = resolved
    _WIKI_CONFIG_FILE.write_text(_json.dumps(cfg, indent=2))
    click.echo(f"Current wiki: {resolved}")


@wiki.command()
@click.argument('path', default='.')
@click.option('--name', '-n', default=None, help='Wiki name')
@click.option('--description', '-d', default='', help='Wiki description')
@click.option('--from-existing', is_flag=True, help='Initialize from directory with existing .md files (moves them to raw/)')
def init(path, name, description, from_existing):
    """Initialize a new wiki at PATH."""
    from underthesea.agent.wiki import WikiAgent
    w = WikiAgent(path)
    if from_existing:
        result = w.init_from_existing(name=name, description=description)
    else:
        result = w.init(name=name or 'My Wiki', description=description)
    click.echo(f"Wiki initialized at {result}")


@wiki.command(name='init-all')
@click.argument('research_dir', default=None, required=False)
def init_all(research_dir):
    """Initialize all subfolders under docs/research/ as wikis.

    Each subfolder with .md files becomes a wiki (existing files move to raw/).
    """
    from underthesea.agent.wiki import WikiAgent
    if research_dir is None:
        research_dir = Path(__file__).resolve().parent.parent / "docs" / "research"
    else:
        research_dir = Path(research_dir)

    if not research_dir.is_dir():
        click.echo(f"Error: {research_dir} not found")
        sys.exit(1)

    count = 0
    for sub in sorted(research_dir.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / ".wiki.json").exists():
            click.echo(f"  skip {sub.name} (already initialized)")
            continue
        w = WikiAgent(sub)
        w.init_from_existing()
        click.echo(f"  init {sub.name}")
        count += 1

    click.echo(f"\n{count} wiki(s) initialized under {research_dir}")


@wiki.command()
@click.argument('source')
@click.option('--wiki-dir', '-w', default=None, help='Wiki directory (overrides current wiki)')
@click.option('--no-compile', is_flag=True, help='Only save to raw/, do not compile with LLM')
def add(source, wiki_dir, no_compile):
    """Add a source (URL or file path) to the wiki.

    \b
    Supported URL types: tweet, arxiv, webpage, github, youtube
    Examples:
      underthesea wiki add https://x.com/karpathy/status/123
      underthesea wiki add https://arxiv.org/abs/2301.00001
      underthesea wiki add ./paper.md
      underthesea wiki add https://example.com/article --no-compile
    """
    from underthesea.agent.wiki import WikiAgent
    w = WikiAgent(_get_wiki_dir(wiki_dir))
    click.echo(f"Adding {source}...")
    result = w.add(source, compile=not no_compile)
    click.echo(result)


@wiki.command()
@click.argument('source')
@click.option('--wiki-dir', '-w', default=None, help='Wiki directory (overrides current wiki)')
def ingest(source, wiki_dir):
    """Ingest a local file into the wiki."""
    from underthesea.agent.wiki import WikiAgent
    w = WikiAgent(_get_wiki_dir(wiki_dir))
    click.echo(f"Ingesting {source}...")
    result = w.ingest(source)
    click.echo(result)


@wiki.command(name='compile')
@click.option('--wiki-dir', '-w', default=None, help='Wiki directory (overrides current wiki)')
def compile_wiki(wiki_dir):
    """Compile all raw documents into wiki articles."""
    from underthesea.agent.wiki import WikiAgent
    w = WikiAgent(_get_wiki_dir(wiki_dir))
    click.echo("Compiling wiki...")
    result = w.compile()
    click.echo(result)


@wiki.command()
@click.argument('question')
@click.option('--wiki-dir', '-w', default=None, help='Wiki directory (overrides current wiki)')
def ask(question, wiki_dir):
    """Ask a question against the wiki."""
    from underthesea.agent.wiki import WikiAgent
    w = WikiAgent(_get_wiki_dir(wiki_dir))
    result = w.ask(question)
    click.echo(result)


@wiki.command()
@click.option('--wiki-dir', '-w', default=None, help='Wiki directory (overrides current wiki)')
def lint(wiki_dir):
    """Run health checks on the wiki."""
    from underthesea.agent.wiki import WikiAgent
    w = WikiAgent(_get_wiki_dir(wiki_dir))
    click.echo("Running wiki lint...")
    result = w.lint()
    click.echo(result)


@wiki.command()
@click.argument('query')
@click.option('--wiki-dir', '-w', default=None, help='Wiki directory (overrides current wiki)')
def search(query, wiki_dir):
    """Search the wiki (no LLM needed)."""
    from underthesea.agent.wiki import WikiAgent
    w = WikiAgent(_get_wiki_dir(wiki_dir))
    results = w.search(query)
    if not results:
        click.echo("No results found.")
        return
    for r in results:
        click.echo(f"  [{r['score']}] {r['title']}")
        click.echo(f"      {r['path']}")
        if r.get('snippet'):
            click.echo(f"      ...{r['snippet']}...")
        click.echo()


@wiki.command()
@click.option('--wiki-dir', '-w', default=None, help='Wiki directory (overrides current wiki)')
def stats(wiki_dir):
    """Show wiki statistics."""
    from underthesea.agent.wiki import WikiAgent
    w = WikiAgent(_get_wiki_dir(wiki_dir))
    s = w.stats()
    click.echo(f"Wiki: {s['name']}")
    click.echo(f"  Raw documents:  {s['raw_docs']}")
    click.echo(f"  Wiki articles:  {s['wiki_articles']}")
    click.echo(f"  Total words:    {s['total_words']:,}")
    click.echo(f"  Output files:   {s['output_files']}")
    if s['categories']:
        click.echo(f"  Categories:     {', '.join(s['categories'])}")


if __name__ == "__main__":
    main()
