"""Pattern extraction agent for discovering new patterns from labeled data.

This module analyzes existing labels that don't fit known patterns and uses
an LLM to discover new patterns that could be added to the taxonomy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from angelica.llm_client.llm import make_structured_llm


class ExampleReference(BaseModel):
    """Reference to a specific example."""
    
    file_path: str = Field(
        description="Path to the source file"
    )
    doc_id: str = Field(
        description="Document ID for reference"
    )
    project: str = Field(
        description="Project name"
    )


class NewPattern(BaseModel):
    """A newly discovered pattern."""
    
    pattern_name: str = Field(
        description="A concise, descriptive name for the pattern (e.g., 'custom_fixture_loader')"
    )
    pattern_description: str = Field(
        description="A detailed description of what characterizes this pattern"
    )
    pattern_category: str = Field(
        description="The category this pattern belongs to (e.g., 'data_load_mechanism', 'data_cleanup_mechanism')"
    )
    example_count: int = Field(
        description="Number of examples that exhibit this pattern"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Confidence that this is a distinct, useful pattern (0.0-1.0)"
    )
    distinguishing_features: List[str] = Field(
        description="Key features that distinguish this pattern from existing ones"
    )
    code_indicators: List[str] = Field(
        description="Specific code patterns, method calls, or annotations that indicate this pattern"
    )
    example_indices: List[int] = Field(
        default_factory=list,
        description="List of example indices (1-based) that exhibit this pattern"
    )
    example_references: List[ExampleReference] = Field(
        default_factory=list,
        description="References to specific examples exhibiting this pattern (file paths and doc IDs)"
    )


class PatternExtractionResult(BaseModel):
    """Result of pattern extraction analysis."""
    
    new_patterns: List[NewPattern] = Field(
        description="List of newly discovered patterns"
    )
    analysis_summary: str = Field(
        description="Summary of the analysis and findings"
    )
    total_examples_analyzed: int = Field(
        description="Total number of examples analyzed"
    )


PATTERN_EXTRACTION_SYSTEM = """Role: Pattern Discovery Expert

You are analyzing test code examples that were labeled as "does_not_fit_with_any_pattern" 
for the field "{field_name}". Your task is to identify NEW patterns that could be added 
to the existing taxonomy.

EXISTING PATTERNS (for reference):
{existing_patterns}

OUTPUT SCHEMA:
{schema_json}

INSTRUCTIONS:
1. Analyze the provided examples and their reasoning
2. Look for common characteristics across multiple examples
3. Identify patterns that are:
   - Distinct from existing patterns
   - Occur in multiple examples (not one-offs)
   - Have clear, identifiable code indicators
   - Would be useful to add to the taxonomy
4. For each new pattern, provide:
   - A clear, descriptive name
   - Detailed description
   - Distinguishing features
   - Code indicators (method calls, annotations, etc.)
   - example_indices: List the example numbers (1-based) that exhibit this pattern
5. Be conservative: only suggest patterns with high confidence (>0.7)
6. If no clear new patterns emerge, return an empty list

Return a JSON object matching the schema exactly.
"""

PATTERN_EXTRACTION_HUMAN = """Analyze these {count} examples labeled as "does_not_fit_with_any_pattern" for field "{field_name}":

{examples}

Based on the reasoning provided for each example, identify new patterns that should be added to the taxonomy.
"""

PATTERN_CONSOLIDATION_SYSTEM = """Role: Pattern Consolidation Expert

You are reviewing a list of patterns discovered from analyzing test code examples.
Your task is to CONSOLIDATE and REFINE these patterns into a smaller, high-quality set.

EXISTING PATTERNS (for reference):
{existing_patterns}

DISCOVERED PATTERNS TO CONSOLIDATE:
{discovered_patterns}

OUTPUT SCHEMA:
{schema_json}

INSTRUCTIONS:
1. Review all discovered patterns carefully
2. Identify patterns that are:
   - Duplicates or very similar (merge them by listing ALL their pattern indices)
   - Too specific or one-offs (remove them)
   - Overlapping in scope (consolidate them)
   - Not distinct from existing patterns (remove them)
3. For consolidated patterns:
   - Choose the best name and description
   - Combine distinguishing features and code indicators
   - In example_indices, list ALL pattern numbers (1-based) that should be merged into this consolidated pattern
   - Set confidence score based on combined evidence
4. Only keep patterns that are:
   - Truly distinct and useful
   - Have strong evidence (multiple examples)
   - Would add value to the taxonomy
5. Aim to reduce the number of patterns by 50-70% through intelligent consolidation
6. IMPORTANT: Use example_indices to indicate which input patterns (by number) are being consolidated

Return a JSON object matching the schema exactly.
"""

PATTERN_CONSOLIDATION_HUMAN = """Consolidate these {count} discovered patterns into a smaller, high-quality set:

Field: {field_name}
Total examples analyzed: {total_examples}

Review the patterns and consolidate them intelligently. For each consolidated pattern, use example_indices to list which input pattern numbers are being merged.
"""


@ray.remote
def process_chunk_remote(
    chunk: List[Dict[str, Any]],
    chunk_idx: int,
    field_name: str,
    existing_patterns: str,
    reasoning_field: str,
    model_env: str,
    temperature: float,
) -> Dict[str, Any]:
    """Ray remote function to process a chunk of examples.
    
    Returns a dict with 'patterns' and 'summary' keys.
    """
    # Create LLM for this worker
    llm = make_structured_llm(
        model_env=model_env,
        schema=PatternExtractionResult,
        temperature=temperature
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", PATTERN_EXTRACTION_SYSTEM),
        ("human", PATTERN_EXTRACTION_HUMAN),
    ])
    
    schema_json = json.dumps(
        PatternExtractionResult.model_json_schema(),
        indent=2,
        ensure_ascii=False
    )
    
    # Format examples
    formatted = []
    for i, ex in enumerate(chunk, 1):
        code = ex["code"]
        if len(code) > 500:
            code = code[:500] + "\n... (truncated)"
        
        formatted.append(
            f"--- Example {i} ---\n"
            f"File: {ex['source']}\n"
            f"Doc ID: {ex['doc_id']}\n"
            f"Project: {ex['project']}\n"
            f"Reasoning ({reasoning_field}): {ex['reasoning']}\n"
            f"Code snippet:\n{code}\n"
        )
    
    examples_text = "\n".join(formatted)
    
    # Run LLM
    chain = prompt | llm
    result = chain.invoke({
        "field_name": field_name,
        "existing_patterns": existing_patterns,
        "schema_json": schema_json,
        "count": len(chunk),
        "examples": examples_text,
    })
    
    # Convert to dict for serialization
    return {
        "patterns": [p.model_dump() for p in result.new_patterns],
        "summary": result.analysis_summary,
        "chunk_idx": chunk_idx,
    }


class PatternExtractor:
    """Agent that discovers new patterns from labeled data in JSON files."""
    
    def __init__(
        self,
        schema: Type[BaseModel],
        existing_patterns: str,
        model_env: str = "OPENAI_MODEL",
        temperature: float = 0.2,
    ):
        self.schema = schema
        self.existing_patterns = existing_patterns
        
        # Create LLM for pattern extraction
        self.llm = make_structured_llm(
            model_env=model_env,
            schema=PatternExtractionResult,
            temperature=temperature
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PATTERN_EXTRACTION_SYSTEM),
            ("human", PATTERN_EXTRACTION_HUMAN),
        ])
        
        self._schema_json = json.dumps(
            PatternExtractionResult.model_json_schema(),
            indent=2,
            ensure_ascii=False
        )
    
    def extract_patterns_from_directory(
        self,
        json_dir: str,
        field_name: str,
        field_value: str,
        reasoning_field: str,
        max_examples: int = 50,
        chunk_size: int = 20,
        use_ray: bool = True,
    ) -> PatternExtractionResult:
        """Extract new patterns from JSON files in a directory.
        
        Args:
            json_dir: Directory containing labeled JSON files (can have subdirectories)
            field_name: The field to analyze (e.g., "data_load_mechanism")
            field_value: The value to filter by (e.g., "does_not_fit_with_any_pattern")
            reasoning_field: The field containing reasoning (e.g., "data_load_mechanism_reasoning")
            max_examples: Maximum number of examples to analyze
            chunk_size: Number of examples to process per LLM call (to avoid token limits)
            use_ray: Whether to use Ray for parallel processing (default: True)
            
        Returns:
            PatternExtractionResult with discovered patterns
        """
        # Load examples from JSON files
        examples = self._load_examples_from_json(
            json_dir, field_name, field_value, reasoning_field, max_examples
        )
        
        if not examples:
            return PatternExtractionResult(
                new_patterns=[],
                analysis_summary=f"No examples found with {field_name}={field_value}",
                total_examples_analyzed=0
            )
        
        # Decide whether to use Ray based on availability and user preference
        use_parallel = use_ray and RAY_AVAILABLE
        
        if use_parallel:
            print("🚀 Using Ray for parallel chunk processing...")
            return self._extract_patterns_parallel(
                examples, field_name, reasoning_field, chunk_size
            )
        else:
            if use_ray and not RAY_AVAILABLE:
                print("⚠️  Ray not available, falling back to sequential processing...")
            return self._extract_patterns_sequential(
                examples, field_name, reasoning_field, chunk_size
            )
    
    def _extract_patterns_parallel(
        self,
        examples: List[Dict[str, Any]],
        field_name: str,
        reasoning_field: str,
        chunk_size: int,
    ) -> PatternExtractionResult:
        """Extract patterns using Ray for parallel processing."""
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Create chunks
        chunks = []
        for chunk_idx in range(0, len(examples), chunk_size):
            chunk = examples[chunk_idx:chunk_idx + chunk_size]
            chunks.append((chunk, chunk_idx))
        
        total_chunks = len(chunks)
        print(f"Processing {total_chunks} chunks in parallel...")
        
        # Submit all chunks to Ray
        futures = []
        for chunk, chunk_idx in chunks:
            future = process_chunk_remote.remote(
                chunk=chunk,
                chunk_idx=chunk_idx,
                field_name=field_name,
                existing_patterns=self.existing_patterns,
                reasoning_field=reasoning_field,
                model_env="OPENAI_MODEL",
                temperature=0.2,
            )
            futures.append(future)
        
        # Collect results as they complete
        all_patterns: List[NewPattern] = []
        chunk_summaries: List[str] = []
        
        for i, future in enumerate(futures):
            result_dict = ray.get(future)
            chunk_num = i + 1
            print(f"✓ Completed chunk {chunk_num}/{total_chunks}")
            
            # Convert dict back to NewPattern objects
            chunk_idx = result_dict["chunk_idx"]
            for pattern_dict in result_dict["patterns"]:
                pattern = NewPattern(**pattern_dict)
                
                # Add example references based on indices
                valid_indices = [
                    chunk_idx + idx - 1
                    for idx in pattern.example_indices
                    if 0 < idx <= len(chunks[i][0])
                ]
                
                pattern.example_references = [
                    ExampleReference(
                        file_path=examples[idx]["source"],
                        doc_id=examples[idx]["doc_id"],
                        project=examples[idx]["project"]
                    )
                    for idx in valid_indices
                ]
                
                pattern.example_count = len(pattern.example_references)
                all_patterns.append(pattern)
            
            chunk_summaries.append(f"Chunk {chunk_num}: {result_dict['summary']}")
        
        # Merge similar patterns across chunks
        merged_patterns = self._merge_similar_patterns(all_patterns, examples)
        
        print(f"\n🔄 Consolidating {len(merged_patterns)} patterns with LLM...")
        
        # Consolidate patterns using LLM
        consolidated_patterns = self._consolidate_patterns_with_llm(
            merged_patterns, field_name, len(examples)
        )
        
        # Create final result
        final_summary = (
            f"Processed {len(examples)} examples in {len(chunk_summaries)} chunks (parallel).\n"
            f"Found {len(all_patterns)} patterns across chunks.\n"
            f"Merged to {len(merged_patterns)} unique patterns.\n"
            f"Consolidated to {len(consolidated_patterns)} high-quality patterns.\n\n"
            + "\n".join(chunk_summaries)
        )
        
        return PatternExtractionResult(
            new_patterns=consolidated_patterns,
            analysis_summary=final_summary,
            total_examples_analyzed=len(examples)
        )
    
    def _extract_patterns_sequential(
        self,
        examples: List[Dict[str, Any]],
        field_name: str,
        reasoning_field: str,
        chunk_size: int,
    ) -> PatternExtractionResult:
        """Extract patterns sequentially (fallback when Ray is not available)."""
        all_patterns: List[NewPattern] = []
        chunk_summaries: List[str] = []
        
        for chunk_idx in range(0, len(examples), chunk_size):
            chunk = examples[chunk_idx:chunk_idx + chunk_size]
            chunk_num = chunk_idx // chunk_size + 1
            total_chunks = (len(examples) + chunk_size - 1) // chunk_size
            
            print(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} examples)...")
            
            # Format examples for the prompt
            examples_text = self._format_examples(chunk, reasoning_field)
            
            # Run the LLM
            chain = self.prompt | self.llm
            result = chain.invoke({
                "field_name": field_name,
                "existing_patterns": self.existing_patterns,
                "schema_json": self._schema_json,
                "count": len(chunk),
                "examples": examples_text,
            })
            
            # Add example references to each pattern based on the indices provided by the LLM
            # Adjust indices to account for chunk offset
            for pattern in result.new_patterns:
                # Convert 1-based indices to 0-based and adjust for chunk offset
                valid_indices = [
                    chunk_idx + idx - 1
                    for idx in pattern.example_indices
                    if 0 < idx <= len(chunk)
                ]
                
                pattern.example_references = [
                    ExampleReference(
                        file_path=examples[idx]["source"],
                        doc_id=examples[idx]["doc_id"],
                        project=examples[idx]["project"]
                    )
                    for idx in valid_indices
                ]
                
                # Update example_count to match actual number of referenced examples
                pattern.example_count = len(pattern.example_references)
            
            all_patterns.extend(result.new_patterns)
            chunk_summaries.append(f"Chunk {chunk_num}: {result.analysis_summary}")
        
        # Merge similar patterns across chunks
        merged_patterns = self._merge_similar_patterns(all_patterns, examples)
        
        print(f"\n🔄 Consolidating {len(merged_patterns)} patterns with LLM...")
        
        # Consolidate patterns using LLM
        consolidated_patterns = self._consolidate_patterns_with_llm(
            merged_patterns, field_name, len(examples)
        )
        
        # Create final result
        final_summary = (
            f"Processed {len(examples)} examples in {len(chunk_summaries)} chunks (sequential).\n"
            f"Found {len(all_patterns)} patterns across chunks.\n"
            f"Merged to {len(merged_patterns)} unique patterns.\n"
            f"Consolidated to {len(consolidated_patterns)} high-quality patterns.\n\n"
            + "\n".join(chunk_summaries)
        )
        
        return PatternExtractionResult(
            new_patterns=consolidated_patterns,
            analysis_summary=final_summary,
            total_examples_analyzed=len(examples)
        )
    
    def _consolidate_patterns_with_llm(
        self,
        patterns: List[NewPattern],
        field_name: str,
        total_examples: int,
    ) -> List[NewPattern]:
        """Use LLM to consolidate patterns into a smaller, high-quality set.
        
        This is a second pass that reviews all discovered patterns and intelligently
        merges, removes, or refines them to produce a final consolidated set.
        """
        if not patterns:
            return []
        
        # If we have very few patterns, skip consolidation
        if len(patterns) <= 3:
            print(f"Only {len(patterns)} patterns found, skipping consolidation.")
            return patterns
        
        # Format patterns for the LLM
        patterns_text = self._format_patterns_for_consolidation(patterns)
        
        # Create consolidation prompt
        consolidation_prompt = ChatPromptTemplate.from_messages([
            ("system", PATTERN_CONSOLIDATION_SYSTEM),
            ("human", PATTERN_CONSOLIDATION_HUMAN),
        ])
        
        # Run LLM
        chain = consolidation_prompt | self.llm
        result = chain.invoke({
            "field_name": field_name,
            "existing_patterns": self.existing_patterns,
            "discovered_patterns": patterns_text,
            "schema_json": self._schema_json,
            "count": len(patterns),
            "total_examples": total_examples,
        })
        
        # The LLM returns consolidated patterns with example_indices indicating which input patterns to merge
        consolidated_raw = result.new_patterns if hasattr(result, 'new_patterns') else []
        
        # Map example references from input patterns to consolidated patterns
        consolidated = []
        for cons_pattern in consolidated_raw:
            # Get all example references from the input patterns indicated by example_indices
            all_refs: Dict[tuple[str, str], ExampleReference] = {}
            
            for pattern_idx in cons_pattern.example_indices:
                # Convert 1-based to 0-based
                idx = pattern_idx - 1
                if 0 <= idx < len(patterns):
                    input_pattern = patterns[idx]
                    # Add all example references from this input pattern
                    for ref in input_pattern.example_references:
                        key = (ref.file_path, ref.doc_id)
                        all_refs[key] = ref
            
            # Update the consolidated pattern with merged example references
            cons_pattern.example_references = list(all_refs.values())
            cons_pattern.example_count = len(all_refs)
            
            # Clear example_indices as they're no longer needed
            cons_pattern.example_indices = []
            
            consolidated.append(cons_pattern)
        
        print(f"✓ Consolidated from {len(patterns)} to {len(consolidated)} patterns")
        
        return consolidated
    
    def _format_patterns_for_consolidation(
        self,
        patterns: List[NewPattern],
    ) -> str:
        """Format patterns for the consolidation prompt."""
        formatted = []
        
        for i, pattern in enumerate(patterns, 1):
            formatted.append(
                f"--- Pattern {i} ---\n"
                f"Name: {pattern.pattern_name}\n"
                f"Category: {pattern.pattern_category}\n"
                f"Description: {pattern.pattern_description}\n"
                f"Example Count: {pattern.example_count}\n"
                f"Confidence: {pattern.confidence_score:.2f}\n"
                f"Distinguishing Features:\n" +
                "\n".join(f"  - {f}" for f in pattern.distinguishing_features) + "\n"
                f"Code Indicators:\n" +
                "\n".join(f"  - {c}" for c in pattern.code_indicators) + "\n"
                f"Example References: {len(pattern.example_references)} examples\n"
            )
        
        return "\n".join(formatted)
    
    def _merge_similar_patterns(
        self,
        patterns: List[NewPattern],
        all_examples: List[Dict[str, Any]],
    ) -> List[NewPattern]:
        """Merge similar patterns found across different chunks.
        
        Patterns are considered similar if they have:
        - Similar pattern names (fuzzy match)
        - Same category
        - High overlap in code indicators
        
        Args:
            patterns: List of patterns from all chunks
            all_examples: All examples for reference mapping
            
        Returns:
            List of merged unique patterns
        """
        if not patterns:
            return []
        
        merged: List[NewPattern] = []
        used_indices: set[int] = set()
        
        for i, pattern in enumerate(patterns):
            if i in used_indices:
                continue
            
            # Find similar patterns
            similar_indices = [i]
            for j in range(i + 1, len(patterns)):
                if j in used_indices:
                    continue
                
                other = patterns[j]
                
                # Check if patterns are similar
                if self._are_patterns_similar(pattern, other):
                    similar_indices.append(j)
                    used_indices.add(j)
            
            # Merge similar patterns
            if len(similar_indices) == 1:
                merged.append(pattern)
            else:
                merged_pattern = self._merge_pattern_group(
                    [patterns[idx] for idx in similar_indices],
                    all_examples
                )
                merged.append(merged_pattern)
        
        return merged
    
    def _are_patterns_similar(self, p1: NewPattern, p2: NewPattern) -> bool:
        """Check if two patterns are similar enough to merge."""
        # Must be same category
        if p1.pattern_category != p2.pattern_category:
            return False
        
        # Check name similarity (simple word overlap)
        words1 = set(p1.pattern_name.lower().replace('_', ' ').split())
        words2 = set(p2.pattern_name.lower().replace('_', ' ').split())
        name_overlap = len(words1 & words2) / max(len(words1), len(words2))
        
        # Check code indicator overlap
        indicators1 = set(p1.code_indicators)
        indicators2 = set(p2.code_indicators)
        if indicators1 and indicators2:
            indicator_overlap = len(indicators1 & indicators2) / max(len(indicators1), len(indicators2))
        else:
            indicator_overlap = 0.0
        
        # Patterns are similar if they have good name overlap OR good indicator overlap
        return name_overlap > 0.6 or indicator_overlap > 0.5
    
    def _merge_pattern_group(
        self,
        patterns: List[NewPattern],
        all_examples: List[Dict[str, Any]],
    ) -> NewPattern:
        """Merge a group of similar patterns into one."""
        # Use the pattern with highest confidence as base
        base = max(patterns, key=lambda p: p.confidence_score)
        
        # Combine all example references (deduplicate by file_path + doc_id)
        all_refs: Dict[tuple[str, str], ExampleReference] = {}
        for pattern in patterns:
            for ref in pattern.example_references:
                key = (ref.file_path, ref.doc_id)
                all_refs[key] = ref
        
        # Combine distinguishing features and code indicators (deduplicate)
        all_features = set()
        all_indicators = set()
        for pattern in patterns:
            all_features.update(pattern.distinguishing_features)
            all_indicators.update(pattern.code_indicators)
        
        # Create merged pattern
        return NewPattern(
            pattern_name=base.pattern_name,
            pattern_description=base.pattern_description,
            pattern_category=base.pattern_category,
            example_count=len(all_refs),
            confidence_score=sum(p.confidence_score for p in patterns) / len(patterns),
            distinguishing_features=sorted(all_features),
            code_indicators=sorted(all_indicators),
            example_indices=[],  # Not used after merging
            example_references=list(all_refs.values()),
        )
    
    def _load_examples_from_json(
        self,
        json_dir: str,
        field_name: str,
        field_value: str,
        reasoning_field: str,
        max_examples: int,
    ) -> List[Dict[str, Any]]:
        """Load examples from JSON files in a directory.
        
        Expected JSON structure:
        {
            "file_path": {
                "doc_id": "...",
                "decided_by": "...",
                "final_label": {
                    "field_name": "value",
                    "reasoning_field": "reasoning text",
                    ...
                }
            },
            ...
        }
        """
        json_path = Path(json_dir)
        if not json_path.exists():
            raise ValueError(f"Directory does not exist: {json_dir}")
        
        examples = []
        
        # Find all JSON files recursively
        json_files = list(json_path.rglob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                # Structure 1: Direct mapping of file paths to labels
                if isinstance(data, dict):
                    for file_path, entry in data.items():
                        if not isinstance(entry, dict):
                            continue
                        
                        # Get the final label
                        final_label = entry.get("final_label", {})
                        if not isinstance(final_label, dict):
                            continue
                        
                        # Check if this example matches our criteria
                        if final_label.get(field_name) == field_value:
                            # Try to get the code/text
                            # It might be in the entry or we might need to read the actual file
                            code = entry.get("code", "")
                            if not code and Path(file_path).exists():
                                try:
                                    code = Path(file_path).read_text(encoding='utf-8', errors='ignore')
                                except Exception:
                                    code = f"[Could not read file: {file_path}]"
                            
                            reasoning = final_label.get(reasoning_field, "No reasoning provided")
                            doc_id = entry.get("doc_id", "unknown")
                            
                            examples.append({
                                "source": file_path,
                                "project": json_file.parent.name,
                                "doc_id": str(doc_id),
                                "code": code,
                                "reasoning": reasoning,
                                "field_value": field_value,
                            })
                            
                            if len(examples) >= max_examples:
                                return examples
                
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON file {json_file}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing {json_file}: {e}")
                continue
        
        return examples
    
    def _format_examples(
        self,
        examples: List[Dict[str, Any]],
        reasoning_field: str,
    ) -> str:
        """Format examples for the prompt."""
        formatted = []
        
        for i, ex in enumerate(examples, 1):
            # Truncate code to first 500 chars for readability
            code = ex["code"]
            if len(code) > 500:
                code = code[:500] + "\n... (truncated)"
            
            formatted.append(
                f"--- Example {i} ---\n"
                f"File: {ex['source']}\n"
                f"Doc ID: {ex['doc_id']}\n"
                f"Project: {ex['project']}\n"
                f"Reasoning ({reasoning_field}): {ex['reasoning']}\n"
                f"Code snippet:\n{code}\n"
            )
        
        return "\n".join(formatted)


