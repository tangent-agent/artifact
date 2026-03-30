"""Label alteration agent for relabeling specific items based on field conditions.

This module allows you to:
1. Load existing labeled JSONs from a directory
2. Filter items based on specific field values
3. Relabel those items using the existing label-dir and label-unit methodology
4. Save the updated labels back to JSON files
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Type
from dataclasses import dataclass

from pydantic import BaseModel
from tqdm import tqdm

from angelica.agents.agents import LabelerAgent, AdjudicatorAgent
from angelica.agents.system import AgenticLabelingSystem
from angelica.models.config import AgenticConfig, LabelingContext
from angelica.storage.sqlite.store_sqlite import SQLiteStore
from angelica.storage.faiss.vector_faiss import FaissVectorIndex
from angelica.storage.faiss.noop_index import NoOpVectorIndex


@dataclass
class FilterCondition:
    """Condition for filtering items to relabel."""
    
    field_name: str
    field_value: Any
    description: str = ""
    
    def matches(self, label: Dict[str, Any]) -> bool:
        """Check if a label matches this condition."""
        return label.get(self.field_name) == self.field_value


@dataclass
class AlterationResult:
    """Result of a label alteration operation."""
    
    total_items: int
    filtered_items: int
    relabeled_items: int
    failed_items: int
    skipped_items: int
    errors: List[Dict[str, Any]]


class LabelAlterer:
    """Agent that alters existing labels based on field conditions."""
    
    def __init__(
        self,
        config: AgenticConfig,
        context: LabelingContext,
        db_path: str = ":memory:",
        index_path: Optional[str] = None,
        model_env: str = "OPENAI_MODEL",
        temperature: float = 0.1,
    ):
        """Initialize the label alterer.
        
        Args:
            config: AgenticConfig with schema, prompts, and patterns
            context: LabelingContext with analysis and project info
            db_path: Path to SQLite database (default: in-memory)
            index_path: Path to FAISS index (optional, for RAG)
            model_env: Environment variable for model selection
            temperature: LLM temperature
        """
        self.config = config
        self.context = context
        self.model_env = model_env
        self.temperature = temperature
        
        # Initialize storage
        self.store = SQLiteStore(db_path=db_path, schema=config.schema)
        
        # Initialize vector index (RAG)
        if index_path and config.enable_rag:
            self.index = FaissVectorIndex(index_dir=index_path)
        else:
            self.index = NoOpVectorIndex()
        
        # Initialize agents
        self.labeler_a = LabelerAgent(
            agent_id="labeler_a",
            schema=config.schema,
            prompt=config.labeler_a_prompt,
            patterns=config.patterns,
            store=self.store,
            index=self.index,
            model_env=model_env,
            temperature=temperature,
            examples_formatter=config.examples_formatter,
        )
        
        self.labeler_b = LabelerAgent(
            agent_id="labeler_b",
            schema=config.schema,
            prompt=config.labeler_b_prompt,
            patterns=config.patterns,
            store=self.store,
            index=self.index,
            model_env=model_env,
            temperature=temperature,
            examples_formatter=config.examples_formatter,
        )
        
        self.adjudicator = AdjudicatorAgent(
            agent_id="adjudicator",
            schema=config.schema,
            prompt=config.adjudicator_prompt,
            patterns=config.patterns,
            store=self.store,
            index=self.index,
            model_env=model_env,
            temperature=temperature,
            examples_formatter=config.examples_formatter,
        )
        
        # Label equality function
        self.label_equality_fn = config.label_equality_fn or self._default_equality
    
    def _default_equality(self, a: BaseModel, b: BaseModel) -> bool:
        """Default equality check: compare all fields."""
        return a.model_dump() == b.model_dump()
    
    def alter_labels_in_directory(
        self,
        json_dir: str,
        output_dir: str,
        filter_conditions: List[FilterCondition],
        dry_run: bool = False,
        backup: bool = True,
    ) -> AlterationResult:
        """Alter labels in JSON files based on filter conditions.
        
        Args:
            json_dir: Directory containing labeled JSON files
            output_dir: Directory to save updated JSON files
            filter_conditions: List of conditions to filter items for relabeling
            dry_run: If True, don't save changes (just report what would be done)
            backup: If True, create backup of original files
            
        Returns:
            AlterationResult with statistics
        """
        json_path = Path(json_dir)
        output_path = Path(output_dir)
        
        if not json_path.exists():
            raise ValueError(f"Input directory does not exist: {json_dir}")
        
        if not dry_run:
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all JSON files
        json_files = list(json_path.rglob("*.json"))
        
        total_items = 0
        filtered_items = 0
        relabeled_items = 0
        failed_items = 0
        skipped_items = 0
        errors = []
        
        print(f"Found {len(json_files)} JSON files to process")
        
        for json_file in tqdm(json_files, desc="Processing files"):
            try:
                # Load JSON file
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, dict):
                    print(f"Warning: Skipping {json_file} - not a dict")
                    continue
                
                # Track if any changes were made
                changes_made = False
                
                # Process each item in the JSON
                for file_path, entry in data.items():
                    total_items += 1
                    
                    if not isinstance(entry, dict):
                        continue
                    
                    final_label = entry.get("final_label", {})
                    if not isinstance(final_label, dict):
                        continue
                    
                    # Check if this item matches any filter condition
                    matches_filter = any(
                        condition.matches(final_label)
                        for condition in filter_conditions
                    )
                    
                    if not matches_filter:
                        skipped_items += 1
                        continue
                    
                    filtered_items += 1
                    
                    # Relabel this item
                    try:
                        new_label = self._relabel_item(entry, file_path)
                        
                        if new_label:
                            # Update the entry with new label
                            entry["final_label"] = new_label.model_dump()
                            entry["decided_by"] = "label_alterer"
                            entry["relabeled"] = True
                            changes_made = True
                            relabeled_items += 1
                        else:
                            failed_items += 1
                            errors.append({
                                "file": str(json_file),
                                "item": file_path,
                                "error": "Relabeling returned None"
                            })
                    
                    except Exception as e:
                        failed_items += 1
                        errors.append({
                            "file": str(json_file),
                            "item": file_path,
                            "error": str(e)
                        })
                        print(f"Error relabeling {file_path}: {e}")
                
                # Save updated JSON if changes were made
                if changes_made and not dry_run:
                    # Create output file path (preserve directory structure)
                    rel_path = json_file.relative_to(json_path)
                    output_file = output_path / rel_path
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Backup original if requested
                    if backup:
                        backup_file = output_file.with_suffix('.json.bak')
                        if json_file != output_file:
                            import shutil
                            shutil.copy2(json_file, backup_file)
                    
                    # Save updated JSON
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
            
            except Exception as e:
                print(f"Error processing file {json_file}: {e}")
                errors.append({
                    "file": str(json_file),
                    "error": str(e)
                })
        
        return AlterationResult(
            total_items=total_items,
            filtered_items=filtered_items,
            relabeled_items=relabeled_items,
            failed_items=failed_items,
            skipped_items=skipped_items,
            errors=errors,
        )
    
    def _relabel_item(
        self,
        entry: Dict[str, Any],
        file_path: str,
    ) -> Optional[BaseModel]:
        """Relabel a single item using the label-dir and label-unit methodology.
        
        Args:
            entry: The JSON entry containing the item to relabel
            file_path: Path to the source file
            
        Returns:
            New label as a Pydantic model, or None if relabeling failed
        """
        # Get the code/content to relabel
        # Try multiple sources for the code
        code = None
        
        # 1. Check if code is in the entry
        if "code" in entry:
            code = entry["code"]
        
        # 2. Try to read from file path
        if not code and Path(file_path).exists():
            try:
                code = Path(file_path).read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                print(f"Warning: Could not read file {file_path}: {e}")
        
        # 3. Try to build from unit_id if using unit-based labeling
        if not code and self.config.unit_resolver and "unit_id" in entry:
            try:
                from angelica.models.config import LabelingUnit
                unit_type = entry.get("unit_type", "method")
                unit_id = entry["unit_id"]
                source = entry.get("source", file_path)
                
                unit = LabelingUnit(
                    unit_type=unit_type,
                    unit_id=unit_id,
                    source=source,
                )
                
                built_doc = self.config.unit_resolver(unit, self.context)
                code = built_doc.content
            except Exception as e:
                print(f"Warning: Could not resolve unit {entry.get('unit_id')}: {e}")
        
        if not code:
            print(f"Warning: No code found for {file_path}")
            return None
        
        # Use the agentic labeling system to relabel
        # Get labels from both labelers
        label_a, _ = self.labeler_a.label(code, current_doc_id=None, k=self.config.examples_k)
        label_b, _ = self.labeler_b.label(code, current_doc_id=None, k=self.config.examples_k)
        
        # Check if they agree
        if self.label_equality_fn(label_a, label_b):
            return label_a
        
        # If they disagree, use adjudicator
        final_label, _ = self.adjudicator.decide(
            code, label_a, label_b,
            current_doc_id=None,
            k=self.config.examples_k
        )
        
        return final_label


def create_alterer_from_config(
    config: AgenticConfig,
    context: LabelingContext,
    db_path: str = ":memory:",
    index_path: Optional[str] = None,
) -> LabelAlterer:
    """Factory function to create a LabelAlterer from config.
    
    Args:
        config: AgenticConfig with schema, prompts, and patterns
        context: LabelingContext with analysis and project info
        db_path: Path to SQLite database
        index_path: Path to FAISS index (optional)
        
    Returns:
        Configured LabelAlterer instance
    """
    return LabelAlterer(
        config=config,
        context=context,
        db_path=db_path,
        index_path=index_path,
    )


