from __future__ import annotations

import shutil

"""Typer-based command line interface.

This CLI supports two workflows:

1) label-dir
   - File-based labeling: reads files and labels each file's content.

2) label-units
   - Unit-based labeling: enumerates units using CONFIG.unit_enumerator(context)
     and resolves each unit into a BuiltDocument using CONFIG.unit_resolver(unit, context).
   - This enables method/class labeling, labeling by FQCN, etc.
   - The analysis object (if needed) is supplied via LabelingContext.analysis.

To keep this library generic, we do NOT import any particular analysis library here by default.
If you want the CLI to construct an analysis object, provide --analysis-provider pointing to a
python file that defines:

    def build_analysis(project_path: str) -> Any

or

    ANALYSIS_PROVIDER: Callable[[str], Any]

Your config's unit_enumerator/unit_resolver can then use ctx.analysis.
"""

import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import typer
from dotenv import load_dotenv
from tqdm import tqdm

from angelica.agents.system import AgenticLabelingSystem
from angelica.metrics.metrics import plot_kappa, rolling_kappa_for_fields
from angelica.models.config import LabelingContext
from angelica.storage.faiss.vector_faiss import FaissVectorIndex
from angelica.storage.faiss.enhanced_vector_faiss import EnhancedFaissVectorIndex
from angelica.storage.sqlite.store_sqlite import SQLiteStore
from angelica.llm_client.token_counter import get_token_counter


def iter_files(path: Path, suffix: str) -> Iterator[Path]:
    """Yield all files under `path` (recursively) that end with `suffix`."""
    if path.is_file() and path.suffix.lower() == suffix.lower():
        yield path
        return
    for p in path.rglob(f"*{suffix}"):
        if p.is_file():
            yield p


def _load_module(path: str, module_name: str):
    p = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def load_config_module(path: str):
    """Dynamically import a python module that defines CONFIG (AgenticConfig)."""
    mod = _load_module(path, "agentic_config_module")
    if not hasattr(mod, "CONFIG"):
        raise RuntimeError("Config module must define CONFIG (AgenticConfig instance).")
    return mod.CONFIG


def load_analysis_provider(path: str) -> Callable[[str], Any]:
    """Load an analysis provider from a python file.

    The provider module must define either:
      - build_analysis(project_path: str) -> Any
      - ANALYSIS_PROVIDER: Callable[[str], Any
    """
    mod = _load_module(path, "analysis_provider_module")
    if hasattr(mod, "build_analysis") and callable(mod.build_analysis):
        return mod.build_analysis
    if hasattr(mod, "ANALYSIS_PROVIDER") and callable(mod.ANALYSIS_PROVIDER):
        return mod.ANALYSIS_PROVIDER
    raise RuntimeError(
        "Analysis provider must define build_analysis(project_path) or ANALYSIS_PROVIDER callable."
    )


app = typer.Typer(
    help="Agentic labeling CLI",
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    add_completion=False,
)


@app.callback()
def main(
    ctx: typer.Context,
    env: str = typer.Option(".env", "--env", help="Path to .env file"),
    temp: float = typer.Option(0.1, "--temp", help="LLM temperature"),
):
    """Global CLI options shared by all commands."""
    load_dotenv(env)
    ctx.obj = {"temp": temp}


@app.command("label-dir")
def label_dir(
    ctx: typer.Context,
    config: str = typer.Option(..., "--config", help="Path to python config file defining CONFIG (AgenticConfig)"),
    path: str = typer.Option(..., "--path", help="Directory (or file) to label"),
    suffix: str = typer.Option(".java", "--suffix", help="File suffix to scan (default: .java)"),
    db: str = typer.Option("labels.db", "--db", help="SQLite database path"),
    index_dir: str = typer.Option("vector_index", "--index-dir", help="FAISS index directory"),
    agent_a: str = typer.Option("labeler_A", "--agent-a", help="Agent id for labeler A"),
    agent_b: str = typer.Option("labeler_B", "--agent-b", help="Agent id for labeler B"),
    adjudicator: str = typer.Option("adjudicator_1", "--adjudicator", help="Agent id for adjudicator"),
    out: Optional[str] = typer.Option(None, "--out", help="Optional output JSON path"),
    fresh_build: bool = typer.Option(False, "--fresh-build", help="If set, then vector index and database will be rebuilt"),
    enhanced: bool = typer.Option(False, "--enhanced", help="Use enhanced system with pattern learning"),
    similarity_threshold: float = typer.Option(0.7, "--similarity-threshold", help="Similarity threshold for enhanced mode"),
    pattern_confidence_threshold: float = typer.Option(0.6, "--pattern-confidence-threshold", help="Pattern confidence threshold for enhanced mode"),
    parallel: bool = typer.Option(False, "--parallel", help="Enable parallel processing with Ray"),
    num_workers: int = typer.Option(4, "--num-workers", help="Number of parallel workers (only with --parallel)"),
    batch_size: int = typer.Option(10, "--batch-size", help="Batch size for FAISS index updates (only with --parallel)"),
    rate_limit_rpm: Optional[int] = typer.Option(None, "--rate-limit-rpm", help="Rate limit in requests per minute (only with --parallel)"),
):
    if fresh_build:
        if Path(index_dir).exists():
            shutil.rmtree(index_dir)
        if Path(db).exists():
            Path(db).unlink()
    temp = ctx.obj["temp"]
    cfg = load_config_module(config)

    root = Path(path).expanduser().resolve()
    files = list(iter_files(root, suffix))

    results: dict[str, Any] = {}
    
    # Use enhanced system if requested
    if enhanced:
        typer.echo("Using enhanced system with pattern learning")
        # Enhanced mode not yet implemented for parallel
        return
    
    # Parallel mode
    if parallel:
        from angelica.parallel import RayLabelingOrchestrator
        
        typer.echo(f"Using parallel mode with {num_workers} workers")
        
        orchestrator = RayLabelingOrchestrator(
            db_path=db,
            index_dir=index_dir,
            config=cfg,
            context=LabelingContext(),
            agent_a_id=agent_a,
            agent_b_id=agent_b,
            adjudicator_id=adjudicator,
            temperature=temp,
            num_workers=num_workers,
            batch_size=batch_size,
            rate_limit_rpm=rate_limit_rpm,
        )
        
        # Prepare documents
        documents: list[tuple[str, Optional[str]]] = [(fp.read_text(encoding="utf-8", errors="ignore"), str(fp)) for fp in files]
        
        # Label in parallel
        parallel_results = orchestrator.label_documents_parallel(documents, show_progress=True)
        
        # Convert to output format
        for fp, res in zip(files, parallel_results):
            results[str(fp)] = {
                "doc_id": res.doc_id,
                "decided_by": res.decided_by,
                "final_label": res.final_label.model_dump(),
            }
        
        # Cleanup
        orchestrator.shutdown()
    
    # Sequential mode (default)
    else:
        store = SQLiteStore(db, schema=cfg.schema, store_spec=cfg.store_spec)
        index = FaissVectorIndex(index_dir)
        system = AgenticLabelingSystem(
            store=store,
            index=index,
            config=cfg,
            context=LabelingContext(),
            agent_a_id=agent_a,
            agent_b_id=agent_b,
            adjudicator_id=adjudicator,
            temperature=temp,
        )

        for fp in tqdm(files, desc="Labeling files"):
            code = fp.read_text(encoding="utf-8", errors="ignore")
            res = system.label_document(code, source=str(fp))
            results[str(fp)] = {
                "doc_id": res.doc_id,
                "decided_by": res.decided_by,
                "final_label": res.final_label.model_dump(),
                "label_a": res.label_a.model_dump(),
                "label_b": res.label_b.model_dump(),
                "prompts": {
                    "labeler_a": res.prompt_a,
                    "labeler_b": res.prompt_b,
                    "adjudicator": res.prompt_adjudicator,
                },
            }

    if out:
        Path(out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        typer.echo(json.dumps(results, indent=2))
    
    # Print token usage summary
    get_token_counter().print_summary()


@app.command("label-units")
def label_units(
    ctx: typer.Context,
    config: str = typer.Option(..., "--config", help="Path to python config file defining CONFIG (AgenticConfig)"),
    db: str = typer.Option("labels.db", "--db", help="SQLite database path"),
    index_dir: str = typer.Option("vector_index", "--index-dir", help="FAISS index directory"),
    agent_a: str = typer.Option("labeler_A", "--agent-a", help="Agent id for labeler A"),
    agent_b: str = typer.Option("labeler_B", "--agent-b", help="Agent id for labeler B"),
    adjudicator: str = typer.Option("adjudicator_1", "--adjudicator", help="Agent id for adjudicator"),
    project_path: Optional[str] = typer.Option(None, "--project-path", help="Optional project root path"),
    analysis_provider: Optional[str] = typer.Option(
        None,
        "--analysis-provider",
        help="Optional python file defining build_analysis(project_path)->analysis",
    ),
    out: Optional[str] = typer.Option(None, "--out", help="Optional output JSON path"),
    fresh_build: bool = typer.Option(False, "--fresh-build", help="If set, then vector index and database will be rebuilt"),
    enhanced: bool = typer.Option(False, "--enhanced", help="Use enhanced system with pattern learning"),
    similarity_threshold: float = typer.Option(0.7, "--similarity-threshold", help="Similarity threshold for enhanced mode"),
    pattern_confidence_threshold: float = typer.Option(0.6, "--pattern-confidence-threshold", help="Pattern confidence threshold for enhanced mode"),
    parallel: bool = typer.Option(False, "--parallel", help="Enable parallel processing with Ray"),
    num_workers: int = typer.Option(4, "--num-workers", help="Number of parallel workers (only with --parallel)"),
    batch_size: int = typer.Option(10, "--batch-size", help="Batch size for FAISS index updates (only with --parallel)"),
    rate_limit_rpm: Optional[int] = typer.Option(None, "--rate-limit-rpm", help="Rate limit in requests per minute (only with --parallel)"),
):
    if fresh_build:
        if Path(index_dir).exists():
            shutil.rmtree(index_dir)
        if Path(db).exists():
            Path(db).unlink()
    """Label units produced by CONFIG.unit_enumerator (method/class/etc.)."""
    temp = ctx.obj["temp"]
    cfg = load_config_module(config)

    if cfg.unit_enumerator is None:
        raise typer.BadParameter("CONFIG.unit_enumerator is not set.")
    if cfg.unit_resolver is None:
        raise typer.BadParameter("CONFIG.unit_resolver is not set.")

    # Build context (analysis is optional; your config may call ctx.require_analysis()).
    ctx_obj = LabelingContext(project_path=project_path)

    if analysis_provider:
        provider = load_analysis_provider(analysis_provider)
        if not project_path:
            raise typer.BadParameter("--project-path is required when using --analysis-provider")
        ctx_obj.analysis = provider(project_path)

    results: dict[str, Any] = {}
    units = list(cfg.unit_enumerator(ctx_obj))
    
    # Use enhanced system if requested
    if enhanced:
        typer.echo("Using enhanced system with pattern learning")
        # Enhanced mode not yet implemented for parallel
        return
    
    # Parallel mode
    if parallel:
        from angelica.parallel import RayLabelingOrchestrator
        
        typer.echo(f"Using parallel mode with {num_workers} workers")
        
        orchestrator = RayLabelingOrchestrator(
            db_path=db,
            index_dir=index_dir,
            config=cfg,
            context=ctx_obj,
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
        for u, res in zip(units, parallel_results):
            results[u.unit_id] = {
                "doc_id": res.doc_id,
                "decided_by": res.decided_by,
                "final_label": res.final_label.model_dump(),
                # "label_a": res.label_a.model_dump(),
                # "label_b": res.label_b.model_dump(),
                "prompts": {
                    "labeler_a": res.prompt_a,
                    "labeler_b": res.prompt_b,
                    "adjudicator": res.prompt_adjudicator,
                },
            }
        
        # Cleanup
        orchestrator.shutdown()
    
    # Sequential mode (default)
    else:
        store = SQLiteStore(db, schema=cfg.schema, store_spec=cfg.store_spec)
        index = FaissVectorIndex(index_dir)
        system = AgenticLabelingSystem(
            store=store,
            index=index,
            config=cfg,
            context=ctx_obj,
            agent_a_id=agent_a,
            agent_b_id=agent_b,
            adjudicator_id=adjudicator,
            temperature=temp,
        )

        for u in tqdm(units, desc="Labeling units"):
            res = system.label_unit(u)
            results[u.unit_id] = {
                "doc_id": res.doc_id,
                "decided_by": res.decided_by,
                "final_label": res.final_label.model_dump(),
                # "label_a": res.label_a.model_dump(),
                # "label_b": res.label_b.model_dump(),
                "prompts": {
                    "labeler_a": res.prompt_a,
                    "labeler_b": res.prompt_b,
                    "adjudicator": res.prompt_adjudicator,
                },
            }

    if out:
        Path(out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        typer.echo(json.dumps(results, indent=2))
    
    # Print token usage summary
    get_token_counter().print_summary()


@app.command("plot-kappa")
def plot_kappa_cmd(
    config: str = typer.Option(..., "--config"),
    db: str = typer.Option("labels.db", "--db"),
    agent_a: str = typer.Option("labeler_A", "--agent-a"),
    agent_b: str = typer.Option("labeler_B", "--agent-b"),
    fields: list[str] = typer.Option(
        ..., "--field", help="Repeatable. E.g. --field pattern_name --field fit_assessment"
    ),
    window: int = typer.Option(50, "--window"),
    no_filter: bool = typer.Option(
        False, "--no-filter", help="Disable filtering by target combinations (TFT, TTT, TTF)"
    ),
):
    """
    Plot rolling kappa for one or more fields.
    
    By default, only tests matching these combinations are included:
    - TFT: is_integration_test=True, is_self_contained=False, is_deployed=True
    - TTT: is_integration_test=True, is_self_contained=True, is_deployed=True
    - TTF: is_integration_test=True, is_self_contained=True, is_deployed=False
    
    Use --no-filter to include all tests.
    """
    cfg = load_config_module(config)
    store = SQLiteStore(db, schema=cfg.schema, store_spec=cfg.store_spec)

    # By default, filter to TFT, TTT, TTF combinations
    # If --no-filter is set, pass None to disable filtering
    target_combos = None if no_filter else None  # Will use DEFAULT_TARGET_COMBINATIONS from function default
    
    df = rolling_kappa_for_fields(
        store=store,
        agent_a=agent_a,
        agent_b=agent_b,
        fields=fields,
        window=window,
        drop_missing=True,
        **({"target_combinations": None} if no_filter else {}),
    )

    typer.echo(df.tail(20).to_string(index=False))
    plot_kappa(df, title=f"{agent_a} vs {agent_b} rolling kappa (window={window})")


@app.command("extract-patterns")
def extract_patterns(
    ctx: typer.Context,
    config: str = typer.Option(..., "--config", help="Path to python config file defining CONFIG (AgenticConfig)"),
    json_dir: str = typer.Option(..., "--json-dir", help="Directory containing labeled JSON files from multiple projects"),
    field_name: str = typer.Option(..., "--field", help="Field to analyze (e.g., data_load_mechanism)"),
    field_value: str = typer.Option(
        "does_not_fit_with_any_pattern",
        "--value",
        help="Value to filter by (default: does_not_fit_with_any_pattern)"
    ),
    reasoning_field: str = typer.Option(..., "--reasoning-field", help="Reasoning field name (e.g., data_load_mechanism_reasoning)"),
    max_examples: int = typer.Option(50, "--max-examples", help="Maximum number of examples to analyze"),
    chunk_size: int = typer.Option(20, "--chunk-size", help="Number of examples per LLM call (to avoid token limits)"),
    use_ray: bool = typer.Option(True, "--use-ray/--no-ray", help="Use Ray for parallel processing (default: True)"),
    out: Optional[str] = typer.Option(None, "--out", help="Optional output JSON path for results"),
):
    """Extract new patterns from labeled JSON files that don't fit existing patterns.
    
    This command analyzes examples labeled with a specific value (typically
    "does_not_fit_with_any_pattern") from JSON files across multiple projects
    and uses an LLM to discover new patterns that could be added to your taxonomy.
    
    The JSON files should contain labeling results in the format:
    {
        "file_path": {
            "final_label": {
                "field_name": "value",
                "reasoning_field": "reasoning text",
                ...
            }
        }
    }
    
    Example:
        angelica extract-patterns \\
            --config coaster_label/config/coaster_config.py \\
            --json-dir coaster_label/config/output \\
            --field data_load_mechanism \\
            --value does_not_fit_with_any_pattern \\
            --reasoning-field data_load_mechanism_reasoning \\
            --max-examples 50 \\
            --out new_patterns.json
    """
    # Import using importlib since directory has hyphen
    import importlib
    pattern_module = importlib.import_module("angelica.post_labeling.new_pattern_extraction")
    PatternExtractor = pattern_module.PatternExtractor
    
    temp = ctx.obj["temp"]
    cfg = load_config_module(config)
    
    typer.echo(f"🔍 Analyzing {field_name}={field_value}")
    typer.echo(f"📊 Using reasoning from: {reasoning_field}")
    typer.echo(f"📁 JSON directory: {json_dir}")
    typer.echo(f"🎯 Max examples: {max_examples}\n")
    
    # Create pattern extractor
    extractor = PatternExtractor(
        schema=cfg.schema,
        existing_patterns=cfg.patterns,
        temperature=temp,
    )
    
    # Extract patterns
    typer.echo("🤖 Running pattern extraction with LLM...\n")
    result = extractor.extract_patterns_from_directory(
        json_dir=json_dir,
        field_name=field_name,
        field_value=field_value,
        reasoning_field=reasoning_field,
        max_examples=max_examples,
        chunk_size=chunk_size,
        use_ray=use_ray,
    )
    
    # Display results
    typer.echo("=" * 80)
    typer.echo("PATTERN EXTRACTION RESULTS")
    typer.echo("=" * 80)
    typer.echo(f"\n📈 Total examples analyzed: {result.total_examples_analyzed}")
    typer.echo(f"🆕 New patterns discovered: {len(result.new_patterns)}\n")
    
    typer.echo("📝 Analysis Summary:")
    typer.echo(result.analysis_summary)
    typer.echo()
    
    if result.new_patterns:
        typer.echo("=" * 80)
        typer.echo("DISCOVERED PATTERNS")
        typer.echo("=" * 80)
        
        for i, pattern in enumerate(result.new_patterns, 1):
            typer.echo(f"\n🔹 Pattern {i}: {pattern.pattern_name}")
            typer.echo(f"   Category: {pattern.pattern_category}")
            typer.echo(f"   Confidence: {pattern.confidence_score:.2f}")
            typer.echo(f"   Examples: {pattern.example_count}")
            typer.echo(f"\n   Description:")
            typer.echo(f"   {pattern.pattern_description}")
            typer.echo(f"\n   Distinguishing Features:")
            for feature in pattern.distinguishing_features:
                typer.echo(f"   • {feature}")
            typer.echo(f"\n   Code Indicators:")
            for indicator in pattern.code_indicators:
                typer.echo(f"   • {indicator}")
            
            # Display example references if available
            if pattern.example_references:
                typer.echo(f"\n   📋 Example References (for verification):")
                for j, ref in enumerate(pattern.example_references[:10], 1):  # Show first 10
                    typer.echo(f"   {j}. File: {ref.file_path}")
                    typer.echo(f"      Doc ID: {ref.doc_id} | Project: {ref.project}")
                if len(pattern.example_references) > 10:
                    typer.echo(f"   ... and {len(pattern.example_references) - 10} more examples")
            typer.echo()
    else:
        typer.echo("ℹ️  No new patterns discovered with sufficient confidence.")
    
    # Save to file if requested
    if out:
        output_data = {
            "field_name": field_name,
            "field_value": field_value,
            "reasoning_field": reasoning_field,
            "total_examples_analyzed": result.total_examples_analyzed,
            "analysis_summary": result.analysis_summary,
            "new_patterns": [p.model_dump() for p in result.new_patterns],
        }
        Path(out).write_text(json.dumps(output_data, indent=2), encoding="utf-8")
        typer.echo(f"\n💾 Results saved to: {out}")
    
    # Print token usage summary
    typer.echo()
    get_token_counter().print_summary()


if __name__ == "__main__":
    app()
