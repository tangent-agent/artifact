"""Pattern extraction agent for discovering new patterns from labeled data.

This module analyzes existing labels that don't fit known patterns and uses
an LLM to discover new patterns that could be added to the taxonomy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

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
5. Be conservative: only suggest patterns with high confidence (>0.7)
6. If no clear new patterns emerge, return an empty list

Return a JSON object matching the schema exactly.
"""

PATTERN_EXTRACTION_HUMAN = """Analyze these {count} examples labeled as "does_not_fit_with_any_pattern" for field "{field_name}":

{examples}

Based on the reasoning provided for each example, identify new patterns that should be added to the taxonomy.
"""


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
    ) -> PatternExtractionResult:
        """Extract new patterns from JSON files in a directory.
        
        Args:
            json_dir: Directory containing labeled JSON files (can have subdirectories)
            field_name: The field to analyze (e.g., "data_load_mechanism")
            field_value: The value to filter by (e.g., "does_not_fit_with_any_pattern")
            reasoning_field: The field containing reasoning (e.g., "data_load_mechanism_reasoning")
            max_examples: Maximum number of examples to analyze
            
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
        
        # Format examples for the prompt
        examples_text = self._format_examples(examples, reasoning_field)
        
        # Run the LLM
        chain = self.prompt | self.llm
        result = chain.invoke({
            "field_name": field_name,
            "existing_patterns": self.existing_patterns,
            "schema_json": self._schema_json,
            "count": len(examples),
            "examples": examples_text,
        })
        
        # Add example references to each pattern
        # For now, add all examples to each pattern since LLM doesn't specify which examples belong to which pattern
        # In a more sophisticated version, we could ask the LLM to map examples to patterns
        for pattern in result.new_patterns:
            pattern.example_references = [
                ExampleReference(
                    file_path=ex["source"],
                    doc_id=ex["doc_id"],
                    project=ex["project"]
                )
                for ex in examples
            ]
        
        return result  # type: ignore[return-value]
    
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


