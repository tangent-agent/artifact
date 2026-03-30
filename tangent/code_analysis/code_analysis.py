from pathlib import Path
from typing import List, Optional, Dict, Iterable

import networkx as nx

from tangent.code_analysis.model.model import PyFunction, PyClass, PyCallSite, PyImport, PyModule, PyApplication
from tangent.code_analysis.backend.codeql import PyCodeQLAnalyzer
from tangent.code_analysis.backend.scalpel import PyScalpelAnalyzer
from tangent.code_analysis.backend.hybrid import PyHybridAnalyzer


class PythonAnalysis:
    """Analysis façade for Python code."""

    def __init__(
            self,
            project_dir: str | Path | None = None,
            source_code: str | None = None,
            analysis_backend_path: str | None = None,
            analysis_json_path: str | Path | None = None,
            eager_analysis: bool = False,
            backend: str = "codeql",
    ) -> None:
        """Initialize Python analysis.

        Args:
            project_dir: Path to the project directory
            source_code: Source code string (alternative to project_dir)
            analysis_backend_path: Path to CodeQL executable (only for codeql backend)
            analysis_json_path: Path to cache analysis results (only for codeql backend)
            eager_analysis: Force re-analysis even if cache exists
            backend: Backend to use ("codeql" or "scalpel")
        """
        self.project_dir = project_dir
        self.source_code = source_code
        self.analysis_json_path = analysis_json_path
        self.analysis_backend_path = analysis_backend_path
        self.eager_analysis = eager_analysis
        self.backend_name = backend

        # Initialize the appropriate backend
        if backend.lower() == "scalpel":
            self.backend = PyScalpelAnalyzer(
                project_dir=project_dir,
                source_code=source_code,
                analysis_json_path=analysis_json_path,
                eager_analysis=eager_analysis,
            )
        elif backend.lower() == "codeql":
            self.backend = PyCodeQLAnalyzer(
                project_dir=project_dir,
                source_code=source_code,
                analysis_backend_path=analysis_backend_path,
                analysis_json_path=analysis_json_path,
                eager_analysis=eager_analysis,
            )
        elif backend.lower() == "hybrid":
            self.backend = PyHybridAnalyzer(
                project_dir=project_dir,
                source_code=source_code,
                analysis_backend_path=analysis_backend_path,
                analysis_json_path=analysis_json_path,
                eager_analysis=eager_analysis,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'codeql' or 'scalpel' or 'hybrid'")

        self._call_graph: nx.DiGraph | None = None

    def get_application_view(self) -> PyApplication:
        """Return the application view of the Python project."""

        return self.backend.get_application_view()

    def get_symbol_table(self) -> Dict[str, PyModule]:
        """Return the symbol table keyed by file path."""

        return self.backend.get_symbol_table()

    def get_modules(self) -> List[PyModule]:
        """Return all modules discovered in the project."""

        app = self.get_application_view()
        return list(app.symbol_table.values())

    # ---------------------------------------------------------------------
    # Java-parity convenience APIs (high-level query helpers)
    # ---------------------------------------------------------------------

    def get_module(self, file_path: str) -> Optional[PyModule]:
        """Return a module by its file-path key (if present)."""

        return self.get_symbol_table().get(file_path)

    def iter_functions(self) -> Iterable[PyFunction]:
        """Iterate over all functions and methods in the application."""

        for mod in self.get_modules():
            for fn in mod.functions:
                yield fn
            for cls in mod.classes:
                for m in cls.methods:
                    yield m

    def get_functions(self) -> List[PyFunction]:
        """Return all *top-level* functions."""

        out: List[PyFunction] = []
        for mod in self.get_modules():
            out.extend(mod.functions)
        return out

    def get_classes(self) -> List[PyClass]:
        """Return all classes."""

        out: List[PyClass] = []
        for mod in self.get_modules():
            out.extend(mod.classes)
        return out

    def get_methods(self) -> List[PyFunction]:
        """Return all methods across all classes."""

        out: List[PyFunction] = []
        for cls in self.get_classes():
            out.extend(cls.methods)
        return out

    def get_imports(self) -> List[PyImport]:
        """Return all import statements."""

        out: List[PyImport] = []
        for mod in self.get_modules():
            out.extend(mod.imports)
        return out

    def get_call_sites(self) -> List[PyCallSite]:
        """Return all call-sites (across functions and methods)."""

        out: List[PyCallSite] = []
        for fn in self.iter_functions():
            out.extend(fn.call_sites)
        return out

    def find_class(self, qualified_name: str) -> Optional[PyClass]:
        """Find a class by its qualified name."""

        for cls in self.get_classes():
            if cls.qualified_name == qualified_name:
                return cls
        return None

    def find_function(self, qualified_name: str) -> Optional[PyFunction]:
        """Find a function/method by its qualified name."""

        for fn in self.iter_functions():
            if fn.qualified_name == qualified_name:
                return fn
        return None

    def get_classes_by_criteria(self, inclusions=None, exclusions=None) -> Dict[str, PyClass]:
        """Return classes filtered by inclusion/exclusion substrings (Java-parity)."""
        inclusions = inclusions or []
        exclusions = exclusions or []
        out: Dict[str, PyClass] = {}
        for cls in self.get_classes():
            name = cls.qualified_name
            if inclusions and not any(s in name for s in inclusions):
                continue
            if exclusions and any(s in name for s in exclusions):
                continue
            out[name] = cls
        return out

    def get_test_methods(self) -> List[PyFunction]:
        """Best-effort: return methods/functions that look like tests."""
        out: List[PyFunction] = []
        for fn in self.iter_functions():
            if fn.name.startswith("test_"):
                out.append(fn)
        # also include methods inside classes flagged as tests (if CodeQL query populates is_test_class)
        for cls in self.get_classes():
            if getattr(cls, "is_test_class", False):
                out.extend(cls.methods)
        # de-dup by qualified name
        seen = set()
        dedup: List[PyFunction] = []
        for f in out:
            if f.qualified_name not in seen:
                seen.add(f.qualified_name)
                dedup.append(f)
        return dedup

    def get_methods_with_decorators(self, decorator_substrings: List[str]) -> List[PyFunction]:
        """Return functions/methods whose decorator expressions contain any substring."""
        out: List[PyFunction] = []
        for fn in self.iter_functions():
            if any(any(s in d.expression for s in decorator_substrings) for d in fn.decorators):
                out.append(fn)
        return out

    def get_calling_lines(self, caller: str) -> List[int]:
        """Return line numbers in *caller* where calls occur (best effort)."""
        fn = self.find_function(caller)
        if fn is None:
            return []
        return [cs.start_line for cs in fn.call_sites if cs.start_line is not None]

    def get_call_targets(self, caller: str) -> List[str]:
        """Return callee identifiers referenced by *caller* call-sites."""
        fn = self.find_function(caller)
        if fn is None:
            return []
        return [cs.method_name for cs in fn.call_sites]

    def get_classes_in_module(self, file_path: str) -> List[PyClass]:
        """Return classes defined in a module (by file path)."""

        mod = self.get_module(file_path)
        return [] if mod is None else list(mod.classes)

    def get_functions_in_module(self, file_path: str) -> List[PyFunction]:
        """Return top-level functions defined in a module (by file path)."""

        mod = self.get_module(file_path)
        return [] if mod is None else list(mod.functions)

    def get_methods_in_module(self, file_path: str) -> List[PyFunction]:
        """Return all methods defined in a module (by file path)."""

        mod = self.get_module(file_path)
        if mod is None:
            return []
        out: List[PyFunction] = []
        for cls in mod.classes:
            out.extend(cls.methods)
        return out

    def get_methods_in_class(self, class_qualified_name: str) -> List[PyFunction]:
        """Return methods for a given class (by qualified name)."""

        cls = self.find_class(class_qualified_name)
        return [] if cls is None else list(cls.methods)

    def get_class(self, qualified_name: str) -> Optional[PyClass]:
        """Alias for :meth:`find_class` (Java-parity)."""

        return self.find_class(qualified_name)

    def get_method(self, qualified_name: str) -> Optional[PyFunction]:
        """Return a method/function by qualified name (Java-parity)."""

        return self.find_function(qualified_name)

    def get_method_parameters(self, qualified_name: str) -> List:
        """Return the parameter list for a function/method."""

        fn = self.find_function(qualified_name)
        return [] if fn is None else list(fn.parameters)

    def get_call_graph_json(self) -> Dict[str, List[str]]:
        """Return call graph as a JSON-serializable adjacency list."""

        g = self.get_call_graph()
        return {n: list(g.successors(n)) for n in g.nodes()}

    def get_callers(self, callee: str) -> List[str]:
        """Return callers of a callee (node id can be qualified or simple)."""

        g = self.get_call_graph()
        return list(g.predecessors(callee)) if callee in g else []

    def get_callees(self, caller: str) -> List[str]:
        """Return callees invoked by a caller (node id can be qualified or simple)."""

        g = self.get_call_graph()
        return list(g.successors(caller)) if caller in g else []

    def get_call_graph(self) -> nx.DiGraph:
        """Return a call graph as a NetworkX directed graph."""

        if self._call_graph is None:
            self._call_graph = self.backend.get_call_graph()
        return self._call_graph
