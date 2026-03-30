#!/usr/bin/env python3
"""
Wrapper script to run tangent labeling with the JSON report.

This script sets up the LabelingContext with the report_path in extras,
then runs the labeling process.

Usage:
    python scripts/run_tangent_labeling.py \\
        --report /path/to/test_modules_report_20260301_225816.json \\
        --db tangent_labels.db \\
        --index-dir tangent_vector_index \\
        --out tangent_results.json \\
        --no-cache  # Optional: disable LLM response caching
"""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent directory to path to import angelica
sys.path.insert(0, str(Path(__file__).parent.parent))

from angelica.agents.system import AgenticLabelingSystem
from angelica.models.config import LabelingContext
from angelica.storage.faiss.vector_faiss import FaissVectorIndex
from angelica.storage.sqlite.store_sqlite import SQLiteStore
from angelica.llm_client.token_counter import get_token_counter
from tangent_label.config.tangent_config import CONFIG


app = typer.Typer(help="Tangent labeling runner")


@app.command()
def label(
    report: str = typer.Option(..., "--report", help="Path to test_modules_report JSON file"),
    db: str = typer.Option("tangent_labels.db", "--db", help="SQLite database path"),
    index_dir: str = typer.Option("tangent_vector_index", "--index-dir", help="FAISS index directory"),
    out: Optional[str] = typer.Option(None, "--out", help="Optional output JSON path"),
    agent_a: str = typer.Option("labeler_A", "--agent-a", help="Agent id for labeler A"),
    agent_b: str = typer.Option("labeler_B", "--agent-b", help="Agent id for labeler B"),
    adjudicator: str = typer.Option("adjudicator_1", "--adjudicator", help="Agent id for adjudicator"),
    temp: float = typer.Option(0.1, "--temp", help="LLM temperature"),
    env_file: str = typer.Option(".env", "--env", help="Path to .env file"),
    fresh_build: bool = typer.Option(False, "--fresh-build", help="Rebuild vector index and database"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable LLM response caching"),
    parallel: bool = typer.Option(False, "--parallel", help="Enable parallel processing with Ray"),
    num_workers: int = typer.Option(4, "--num-workers", help="Number of parallel workers (only with --parallel)"),
    batch_size: int = typer.Option(10, "--batch-size", help="Batch size for FAISS index updates (only with --parallel)"),
    rate_limit_rpm: Optional[int] = typer.Option(None, "--rate-limit-rpm", help="Rate limit in requests per minute (only with --parallel)"),
):
    """Label Python test modules from the tangent JSON report."""
    
    # Load environment
    load_dotenv(env_file)
    
    # Disable cache if requested
    if no_cache:
        import os
        import shutil
        
        # Remove existing cache to avoid constraint errors
        cache_dir = os.getenv("LLM_CACHE_DIR", ".llm_cache")
        if Path(cache_dir).exists():
            shutil.rmtree(cache_dir)
            typer.echo(f"🗑️  Removed existing cache: {cache_dir}")
        
        # Disable cache for this run
        os.environ["ENABLE_LLM_CACHE"] = "false"
        typer.echo("🚫 LLM caching disabled")
        
        # For Ray workers, we need to set this in the Ray runtime environment
        if parallel:
            # Set runtime env for Ray workers
            import ray
            if not ray.is_initialized():
                ray.init(
                    ignore_reinit_error=True,
                    runtime_env={"env_vars": {"ENABLE_LLM_CACHE": "false"}}
                )
    
    # Verify report exists
    report_path = Path(report).expanduser().resolve()
    if not report_path.exists():
        typer.echo(f"❌ Error: Report file not found: {report_path}", err=True)
        raise typer.Exit(1)
    
    # Clean up if fresh build
    if fresh_build:
        import shutil
        if Path(index_dir).exists():
            shutil.rmtree(index_dir)
            typer.echo(f"🗑️  Removed existing index: {index_dir}")
        if Path(db).exists():
            Path(db).unlink()
            typer.echo(f"🗑️  Removed existing database: {db}")
    
    # Create context with report_path in extras
    ctx = LabelingContext(extras={"report_path": str(report_path)})
    
    # Enumerate units
    typer.echo(f"📊 Reading report: {report_path}")
    if CONFIG.unit_enumerator is None:
        typer.echo("❌ Error: CONFIG.unit_enumerator is not set", err=True)
        raise typer.Exit(1)
    units = list(CONFIG.unit_enumerator(ctx))
    typer.echo(f"✅ Found {len(units)} test modules to label\n")
    
    # Parallel mode
    if parallel:
        from angelica.parallel import RayLabelingOrchestrator
        
        typer.echo(f"🚀 Using parallel mode with {num_workers} workers")
        
        orchestrator = RayLabelingOrchestrator(
            db_path=db,
            index_dir=index_dir,
            config=CONFIG,
            context=ctx,
            agent_a_id=agent_a,
            agent_b_id=agent_b,
            adjudicator_id=adjudicator,
            temperature=temp,
            num_workers=num_workers,
            batch_size=batch_size,
            rate_limit_rpm=rate_limit_rpm,
        )
        
        # Label in parallel
        parallel_results = orchestrator.label_units_parallel(units, show_progress=True)
        
        # Convert to output format
        results = {}
        for unit, res in zip(units, parallel_results):
            # Handle error cases where final_label might be None
            final_label_dict = res.final_label.model_dump() if res.final_label else None
            results[unit.unit_id] = {
                "doc_id": res.doc_id,
                "decided_by": res.decided_by,
                "final_label": final_label_dict,
                "source": unit.source,
                "error": res.decided_by == "error",
            }
        
        # Cleanup
        orchestrator.shutdown()
    
    # Sequential mode (default)
    else:
        # Set up storage and system
        store = SQLiteStore(db, schema=CONFIG.schema, store_spec=CONFIG.store_spec)
        index = FaissVectorIndex(index_dir)
        system = AgenticLabelingSystem(
            store=store,
            index=index,
            config=CONFIG,
            context=ctx,
            agent_a_id=agent_a,
            agent_b_id=agent_b,
            adjudicator_id=adjudicator,
            temperature=temp,
        )
        
        # Label each unit
        results = {}
        for unit in tqdm(units, desc="Labeling test modules"):
            res = system.label_unit(unit)
            results[unit.unit_id] = {
                "doc_id": res.doc_id,
                "decided_by": res.decided_by,
                "final_label": res.final_label.model_dump(),
                "source": unit.source,
            }
    
    # Save or print results
    if out:
        Path(out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        typer.echo(f"\n💾 Results saved to: {out}")
    else:
        typer.echo("\n" + "="*80)
        typer.echo("RESULTS")
        typer.echo("="*80)
        typer.echo(json.dumps(results, indent=2))
    
    # Print summary statistics
    typer.echo("\n" + "="*80)
    typer.echo("SUMMARY")
    typer.echo("="*80)
    total = len(results)
    agent_related = sum(1 for r in results.values() 
                       if r["final_label"]["is_agent_or_tool_related_test"])
    typer.echo(f"Total test modules: {total}")
    typer.echo(f"Agent/tool related: {agent_related} ({agent_related/total*100:.1f}%)")
    typer.echo(f"Not agent/tool related: {total - agent_related} ({(total-agent_related)/total*100:.1f}%)")
    
    # Print token usage
    typer.echo()
    get_token_counter().print_summary()


if __name__ == "__main__":
    app()
