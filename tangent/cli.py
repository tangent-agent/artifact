from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console


app = typer.Typer(add_completion=False, help="Tangent: static analysis utilities for agentic repos.")
console = Console()


@app.command("version")
def version() -> None:
    """Print the Tangent version."""
    # Keep a second command so Typer behaves as a command group (so `tangent analyze ...` works).
    console.print("0.0.1")


@app.command("analyze")
def analyze(
    repo: Path = typer.Option(..., "--repo", "-r", exists=True, file_okay=False, dir_okay=True, help="Repo to analyze"),
    out: Path = typer.Option(
        Path("analysis.json"),
        "--out",
        "-o",
        help="Where to write the analysis.json (default: ./analysis.json)",
    ),
    backend: str = typer.Option("cldk", "--backend", help="Analysis backend: 'codeql' or 'cldk' (default: cldk)"),
    codeql: str = typer.Option("codeql", "--codeql", help="Path to the CodeQL CLI executable (only for codeql backend)"),
    keep_workdir: bool = typer.Option(False, "--keep-workdir", help="Keep CodeQL temp dir for debugging (only for codeql backend)"),
    caller_hops: int = typer.Option(1, "--caller-hops", help="Depth of call tree traversal backward from agents (default: 1)"),
    cldk_backend: str = typer.Option("scalpel", "--cldk-backend", help="CLDK backend: 'scalpel' or 'codeql' (default: scalpel, only for cldk backend)"),
):
    """Run static analysis and write an Application-shaped analysis.json.
    
    Supports two backends:
    - 'cldk': Uses CLDK library (faster, no external dependencies)
    - 'codeql': Uses CodeQL queries (more detailed, requires CodeQL CLI)
    """

    try:
        if backend.lower() == "cldk":
            # Use CLDK-based analyzer
            from tangent.agent_analysis.analyzer import analyze_repo
            
            analysis = analyze_repo(
                repo=repo,
                out=out,
                repo_name=repo.name,
                caller_hops=caller_hops,
                backend=cldk_backend,
            )
        elif backend.lower() == "codeql":
            # Use original CodeQL-based analyzer
            from tangent.old_folders.codeql import analyze_repo
            
            analysis = analyze_repo(
                repo=repo,
                out=out,
                repo_name=repo.name,
                codeql=codeql,
                keep_workdir=keep_workdir,
                caller_hops=caller_hops,
            )
        else:
            console.print(f"[red]Unknown backend:[/red] {backend}. Use 'codeql' or 'cldk'")
            raise typer.Exit(code=1)
            
    except Exception as e:
        console.print(f"[red]Analysis failed:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)

    console.print(f"[green]Wrote[/green] {out}")
    console.print(f"Frameworks: {analysis.get('framework', [])}")
    console.print(f"Factory Agents: {len(analysis.get('factory_agents', []))} | Tests: {len(analysis.get('tests', []))}")

if __name__ == "__main__":
    app()
