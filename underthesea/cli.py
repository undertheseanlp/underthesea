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
    version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
    underthesea_version = open(version_file).read().strip()
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


if __name__ == "__main__":
    main()
