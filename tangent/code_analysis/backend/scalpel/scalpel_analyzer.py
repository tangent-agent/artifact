

"""Python Scalpel backend for Tangent code analysis.

This backend uses AST-based analysis for Python static analysis as an
alternative to CodeQL. It provides symbol table extraction and
call graph generation.
"""

from __future__ import annotations

import ast
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List

import networkx as nx
from tqdm import tqdm

from tangent.code_analysis.model.model import (
    PyApplication,
    PyArgument,
    PyCallSite,
    PyClass,
    PyDecorator,
    PyFunction,
    PyImport,
    PyModule,
)

logger = logging.getLogger(__name__)


class PyScalpelAnalyzer:
    """AST-backed analysis builder for Python."""

    def __init__(
        self,
        project_dir: str | Path | None,
        source_code: str | None = None,
        analysis_json_path: str | Path | None = None,
        eager_analysis: bool = False,
        show_progress: bool = True,
    ) -> None:
        if project_dir is None and source_code is None:
            raise ValueError("Either project_dir or source_code must be provided")

        self.project_dir = Path(project_dir) if project_dir else None
        self.source_code = source_code
        self.analysis_json_path = Path(analysis_json_path).resolve() if analysis_json_path else None
        self.eager_analysis = eager_analysis
        self.show_progress = show_progress

        self._app: PyApplication | None = None
        self._call_graph: nx.DiGraph | None = None
        
        # Type inference support
        self._types_index: Dict[tuple, set[str]] = {}
        self._fqn_cache: Dict[str, Dict[str, str]] = {}

        if self.source_code is not None:
            # In single-file mode we write to a temporary directory.
            tmp = Path(tempfile.mkdtemp(prefix="tangent-py-src-"))
            (tmp / "input.py").write_text(self.source_code, encoding="utf-8")
            self.project_dir = tmp

    def _analysis_cache_file(self) -> Path | None:
        if self.analysis_json_path is None:
            return None
        # self.analysis_json_path.mkdir(parents=True, exist_ok=True)
        return self.analysis_json_path 

    def _load_cached(self) -> PyApplication | None:
        cache = self._analysis_cache_file()
        if not cache or not cache.exists() or self.eager_analysis:
            return None
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            return PyApplication(**data)
        except Exception as e:
            logger.warning("Failed to load cached analysis.json (%s); regenerating.", e)
            return None


    def _build_import_symbol_map(self, module: PyModule) -> Dict[str, str]:
        """Build map: symbol name -> fully-qualified name based on module.imports."""
        out: Dict[str, str] = {}
        for imp in module.imports or []:
            frm = (imp.from_statement or "").strip()
            for name in (imp.imports or []):
                n = (name or "").strip()
                if not n:
                    continue
                if frm:
                    out[n] = f"{frm}.{n}"
                else:
                    out[n] = n
        return out

    def _run_scalpel_type_inference(self, module_file: str | Path) -> List[Dict]:
        """Run Scalpel's TypeInference to get type information.
        
        Returns list of type inference records.
        Safe/no-op if TypeInference isn't available.
        """
        try:
            from scalpel.typeinfer.typeinfer import TypeInference
        except Exception:
            return []
        
        import sys
        import warnings
        original_path = sys.path.copy()
        
        try:
            mp = Path(module_file).resolve()
            
            # Remove the project directory from sys.path to avoid naming conflicts
            # with built-in modules (e.g., keyword.py conflicting with keyword module)
            if self.project_dir:
                project_str = str(self.project_dir.resolve())
                sys.path = [p for p in sys.path if not p.startswith(project_str)]
            
            # Suppress warnings and stderr from Scalpel library
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inferer = TypeInference(name=str(mp.stem), entry_point=str(mp))
                inferer.infer_types()
                return inferer.get_types() or []
        except (ImportError, SyntaxError) as e:
            # Skip files with import errors (e.g., circular imports, naming conflicts)
            # or syntax errors - these are common in analyzed repositories
            # Check for known Scalpel compatibility issues
            error_str = str(e)
            if "cannot import name 'deque' from 'collections'" in error_str:
                # Scalpel library has Python 3.11+ compatibility issues
                # Only log once per session
                if not hasattr(self, '_deque_error_logged'):
                    logger.warning("Scalpel library has compatibility issues with Python 3.11+ (deque import error). Type inference will be skipped.")
                    self._deque_error_logged = True
                return []
            logger.debug("Skipping type inference for %s due to %s: %s",
                        module_file, type(e).__name__, error_str[:100])
            return []
        except AttributeError as e:
            # Handle compatibility issues with inspect module
            error_str = str(e)
            if "'cleandoc'" in error_str or "cleandoc" in error_str:
                # Only log once per session
                if not hasattr(self, '_cleandoc_error_logged'):
                    logger.warning("Scalpel library requires Python 3.13+ for full functionality (inspect.cleandoc). Type inference will be limited.")
                    self._cleandoc_error_logged = True
                return []
            logger.debug("Type inference failed for %s: %s", module_file, e)
            return []
        except Exception as e:
            # Log but don't fail - type inference is optional
            logger.debug("Type inference failed for %s: %s", module_file, e)
            return []
        finally:
            # Always restore original sys.path
            sys.path = original_path

    def _run_fqn_inference(self, module_file: str | Path) -> Dict[str, str]:
        """Run Scalpel's FQN inference to get fully qualified names for calls.
        
        Returns a dict mapping call expressions to their FQN.
        Safe/no-op if FQNInference isn't available.
        """
        try:
            from scalpel.fqn.fully_qualified_name_inference import FullyQualifiedNameInference as FQNInference
        except Exception:
            return {}
        
        module_path = Path(module_file).resolve()
        cache_key = str(module_path)
        
        if cache_key in self._fqn_cache:
            return self._fqn_cache[cache_key]
        
        # Common Python built-in functions
        builtin_names = {
            'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
            'open', 'input', 'type', 'isinstance', 'hasattr', 'getattr', 'setattr',
            'enumerate', 'zip', 'map', 'filter', 'sorted', 'sum', 'min', 'max', 'abs',
            'all', 'any', 'bool', 'bytes', 'chr', 'ord', 'dir', 'eval', 'exec', 'format',
            'frozenset', 'hash', 'help', 'hex', 'id', 'iter', 'next', 'object', 'oct',
            'pow', 'repr', 'reversed', 'round', 'slice', 'super', 'vars'
        }
        
        import sys
        import warnings
        original_path = sys.path.copy()
        
        try:
            # Remove the project directory from sys.path to avoid naming conflicts
            # with built-in modules (e.g., keyword.py conflicting with keyword module)
            if self.project_dir:
                project_str = str(self.project_dir.resolve())
                sys.path = [p for p in sys.path if not p.startswith(project_str)]
            
            # Suppress warnings from Scalpel library
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fqn_list = FQNInference(file_path=str(module_path)).infer()
            
            fqn_dict = {}
            
            for fqn in fqn_list:
                fqn_dict[fqn] = fqn
                
                if '.' in fqn:
                    parts = fqn.split('.')
                    if len(parts) >= 2:
                        class_method = '.'.join(parts[-2:])
                        if class_method not in fqn_dict:
                            fqn_dict[class_method] = fqn
                    
                    method_name = parts[-1]
                    if method_name not in fqn_dict:
                        fqn_dict[method_name] = fqn
            
            # Add built-in functions with <builtin> prefix
            for builtin_name in builtin_names:
                builtin_fqn = f"<builtin>.{builtin_name}"
                # Add both the full FQN and the short name mapping
                fqn_dict[builtin_fqn] = builtin_fqn
                # Only map short name if not already mapped to application code
                if builtin_name not in fqn_dict:
                    fqn_dict[builtin_name] = builtin_fqn
            
            self._fqn_cache[cache_key] = fqn_dict
            return fqn_dict
        except (ImportError, SyntaxError) as e:
            # Skip files with import errors (e.g., circular imports, naming conflicts)
            # or syntax errors - these are common in analyzed repositories
            error_str = str(e)
            if "cannot import name 'deque' from 'collections'" in error_str:
                # Scalpel library has Python 3.11+ compatibility issues
                # Errors already logged in type inference, skip silently here
                self._fqn_cache[cache_key] = {}
                return {}
            logger.debug("Skipping FQN inference for %s due to %s: %s",
                        module_file, type(e).__name__, error_str[:100])
            self._fqn_cache[cache_key] = {}
            return {}
        except AttributeError as e:
            # Handle compatibility issues with inspect module
            error_str = str(e)
            if "'cleandoc'" in error_str or "cleandoc" in error_str:
                # Errors already logged in type inference, skip silently here
                self._fqn_cache[cache_key] = {}
                return {}
            logger.debug("FQN inference failed for %s: %s", module_file, e)
            self._fqn_cache[cache_key] = {}
            return {}
        except Exception as e:
            # Log but don't fail - FQN inference is optional
            logger.debug("FQN inference failed for %s: %s", module_file, e)
            self._fqn_cache[cache_key] = {}
            return {}
        finally:
            # Always restore original sys.path
            sys.path = original_path

    def _norm_func_names(self, func: str) -> List[str]:
        """Return possible function name aliases to index/query under."""
        func = (func or "").strip()
        if not func:
            return []
        names = {func}

        if "." in func:
            parts = func.split(".")
            names.add(parts[-1])
            if len(parts) >= 2:
                names.add(".".join(parts[-2:]))

        for sep in ("::", ":"):
            if sep in func:
                parts = [p for p in func.split(sep) if p]
                names.add(parts[-1])
                if len(parts) >= 2:
                    names.add(".".join(parts[-2:]))

        return sorted(names)

    def _norm_abs_file(self, raw_file: str, fallback_abs: str) -> str:
        raw_file = (raw_file or "").strip()
        if not raw_file:
            return str(Path(fallback_abs).resolve())

        p = Path(raw_file)
        if p.is_absolute():
            return str(p.resolve())

        if self.project_dir:
            return str((self.project_dir / p).resolve())

        return str(Path(fallback_abs).resolve())

    def _index_inferred_item(self, item: Dict, fallback_module_abs: str) -> None:
        """Index one Scalpel inferred record under multiple aliases."""
        abs_file = self._norm_abs_file(item.get("file") or "", fallback_module_abs)
        line = int(item.get("line_number") or 0)

        func_raw = str(item.get("function") or "")
        func_names = self._norm_func_names(func_raw)
        if not func_names:
            return

        var = str(item.get("variable") or item.get("parameter") or "__return__")

        ty = item.get("type")
        tys: set[str] = set()
        if isinstance(ty, set):
            tys = {str(x) for x in ty}
        elif isinstance(ty, (list, tuple)):
            tys = {str(x) for x in ty}
        elif isinstance(ty, str):
            tys = {ty}
        elif ty is not None:
            tys = {str(ty)}

        if not tys:
            return

        for fn in func_names:
            self._types_index[(abs_file, fn, var, line)] = tys

    def _nearest_type(
        self,
        *,
        file_path: str,
        function_name: str,
        variable_name: str,
        line_number: int,
    ) -> set[str] | None:
        """Return type(s) for var at/before line_number."""
        abs_file = str(Path(file_path).resolve())
        
        # Try exact match first
        key = (abs_file, function_name, variable_name, line_number)
        if key in self._types_index:
            return self._types_index[key]
        
        # Try earlier lines in same function
        candidates = [
            (k, v) for k, v in self._types_index.items()
            if k[0] == abs_file and k[1] == function_name and k[2] == variable_name and k[3] <= line_number
        ]
        if candidates:
            candidates.sort(key=lambda x: x[0][3], reverse=True)
            return candidates[0][1]
        
        return None
    def _persist(self, app: PyApplication) -> None:
        cache = self._analysis_cache_file()
        if not cache:
            return
        cache.write_text(app.model_dump_json(indent=2), encoding="utf-8")

    def _parse_module(self, file_path: Path) -> PyModule:
        # qualified_name should be relative path from project directory
        if self.project_dir:
            try:
                rel_path = file_path.relative_to(self.project_dir)
                qualified_name = str(rel_path).replace('\\', '/')  # Normalize path separators
            except ValueError:
                # If path is not relative to project_dir, use the file name
                qualified_name = file_path.name
        else:
            qualified_name = file_path.name
        
        abs_file_path = str(file_path)  # Keep absolute path for file_path
        
        try:
            source = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.debug("Skipping file (read error) %s: %s", file_path, e)
            return PyModule(
                file_path=abs_file_path,
                qualified_name=qualified_name,
                is_test=False,
                functions=[],
                classes=[],
                imports=[],
            )

        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            # Syntax errors in analyzed code are common and expected - log at debug level
            logger.debug("Skipping file (syntax error) %s: %s", file_path, e)
            return PyModule(
                file_path=abs_file_path,
                qualified_name=qualified_name,
                is_test=False,
                functions=[],
                classes=[],
                imports=[],
            )

        functions: List[PyFunction] = []
        classes: List[PyClass] = []
        imports: List[PyImport] = []

        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.append(PyImport(from_statement="", imports=[a.name for a in node.names]))
            elif isinstance(node, ast.ImportFrom):
                imports.append(PyImport(from_statement=node.module or "", imports=[a.name for a in node.names]))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(self._build_function(node, file_path.stem))
            elif isinstance(node, ast.ClassDef):
                classes.append(self._build_class(node, file_path.stem))

        # Determine if this is a test module based on qualified_name (relative path).
        # Prefer false positives over false negatives: any path segment or filename
        # that looks test-related is flagged.
        fname = file_path.name
        # Split into directory components (exclude the filename itself)
        path_parts = qualified_name.replace("\\", "/").split("/")
        dir_parts_lower = [p.lower() for p in path_parts[:-1]]  # directory components only
        is_test_module = (
            # Filename checks
            fname.startswith("test_") or
            fname.endswith("_test.py") or
            fname.startswith("spec_") or
            fname.endswith("_spec.py") or
            fname == "conftest.py" or
            # Any directory component contains "test" as a substring
            # (covers test/, tests/, integration_tests/, unit_tests/, etc.)
            any("test" in part for part in dir_parts_lower)
        )

        return PyModule(
            file_path=abs_file_path,  # Keep absolute path
            qualified_name=qualified_name,  # Use relative path
            is_test=is_test_module,
            functions=functions,
            classes=classes,
            imports=imports,
        )

    def _build_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, module: str) -> PyFunction:
        parameters = []
        for arg in node.args.args:
            annotation = None
            if arg.annotation:
                try:
                    annotation = ast.unparse(arg.annotation)
                except Exception:
                    annotation = None
            parameters.append(PyArgument(name=arg.arg, annotation=annotation))

        return_type = None
        if node.returns:
            try:
                return_type = ast.unparse(node.returns)
            except Exception:
                return_type = None

        decorators = []
        for dec in node.decorator_list:
            try:
                decorators.append(PyDecorator(expression=ast.unparse(dec)))
            except Exception:
                pass

        qualified_name = f"{module}.{node.name}"
        
        # Determine if this is a test function based on naming convention or decorators
        is_test_func = (
            node.name.startswith("test_") or
            node.name.endswith("_test") or
            any(self._is_test_decorator(dec) for dec in node.decorator_list)
        )
        
        # Determine if this is an async function
        is_async_func = isinstance(node, ast.AsyncFunctionDef)
        
        return PyFunction(
            qualified_module_name="",  # Will be set from PyModule later
            qualified_name=qualified_name,
            name=node.name,
            kind="function",
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            docstring=ast.get_docstring(node),
            is_test=is_test_func,
            is_async=is_async_func,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            call_sites=[],
        )
    
    def _is_test_decorator(self, decorator: ast.expr) -> bool:
        """Check if a decorator is test-related.

        Errs on the side of inclusion (no false negatives): any decorator
        that mentions common testing keywords is treated as test-related.
        """
        try:
            dec_str = ast.unparse(decorator).lower()
            return (
                "test" in dec_str or
                "pytest" in dec_str or
                "unittest" in dec_str or
                "given" in dec_str or          # Hypothesis: @given(...)
                "parameteriz" in dec_str or    # @parameterized / @parameterize
                "fixture" in dec_str or        # @pytest.fixture
                "mark" in dec_str or           # @pytest.mark.*
                "setup" in dec_str or          # @setup / @setUp variants
                "teardown" in dec_str          # @teardown / @tearDown variants
            )
        except Exception:
            return False

    def _build_class(self, node: ast.ClassDef, module: str) -> PyClass:
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                pass

        decorators = []
        for dec in node.decorator_list:
            try:
                decorators.append(PyDecorator(expression=ast.unparse(dec)))
            except Exception:
                pass

        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_params = []
                for arg in item.args.args:
                    annotation = None
                    if arg.annotation:
                        try:
                            annotation = ast.unparse(arg.annotation)
                        except Exception:
                            annotation = None
                    method_params.append(PyArgument(name=arg.arg, annotation=annotation))

                method_return_type = None
                if item.returns:
                    try:
                        method_return_type = ast.unparse(item.returns)
                    except Exception:
                        method_return_type = None

                method_decorators = []
                for dec in item.decorator_list:
                    try:
                        method_decorators.append(PyDecorator(expression=ast.unparse(dec)))
                    except Exception:
                        pass

                qualified_name = f"{module}.{node.name}.{item.name}"
                
                # Determine if this is a test method.
                # Also flag unittest lifecycle methods (setUp, tearDown, etc.) as
                # test-related so they are never missed (no false negatives).
                _UNITTEST_LIFECYCLE = {
                    "setUp", "tearDown", "setUpClass", "tearDownClass",
                    "setUpModule", "tearDownModule", "addCleanup",
                    "assertRaises", "subTest",
                }
                is_test_method = (
                    item.name.startswith("test_") or
                    item.name.endswith("_test") or
                    item.name in _UNITTEST_LIFECYCLE or
                    any(self._is_test_decorator(dec) for dec in item.decorator_list)
                )
                
                # Determine if this is an async method
                is_async_method = isinstance(item, ast.AsyncFunctionDef)
                
                methods.append(
                    PyFunction(
                        qualified_module_name="",  # Will be set from PyModule later
                        qualified_name=qualified_name,
                        name=item.name,
                        kind="method",
                        parameters=method_params,
                        return_type=method_return_type,
                        decorators=method_decorators,
                        docstring=ast.get_docstring(item),
                        is_test=is_test_method,
                        is_async=is_async_method,
                        start_line=item.lineno,
                        end_line=item.end_lineno or item.lineno,
                        call_sites=[],
                    )
                )

        # A class is a test class if its name follows Test* / *Test convention,
        # OR if it inherits from any base whose name contains "Test" or "TestCase"
        # (covers unittest.TestCase, django.test.TestCase, etc.).
        # Prefer false positives over false negatives.
        base_names = [b.split(".")[-1] for b in bases]  # strip module prefix
        is_test = (
            node.name.startswith("Test") or
            node.name.endswith("Test") or
            any("test" in b.lower() for b in base_names) or
            any("testcase" in b.lower() for b in base_names)
        )

        # If the class is a test class, mark ALL its methods as is_test=True
        # so that setUp/tearDown/helper methods inside test classes are never
        # missed (no false negatives at the method level).
        # Use model_copy() because PyFunction is a Pydantic model.
        if is_test:
            methods = [m.model_copy(update={"is_test": True}) for m in methods]

        return PyClass(
            qualified_name=f"{module}.{node.name}",
            class_name=node.name,
            bases=bases,
            decorators=decorators,
            docstring=ast.get_docstring(node),
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            methods=methods,
            is_test_class=is_test,
        )

    def _attach_callsites(self, module: PyModule) -> None:
        """Extract call sites from the module."""
        try:
            tree = ast.parse(Path(module.file_path).read_text(encoding="utf-8"))
        except SyntaxError as e:
            # Syntax errors are already logged during initial parsing - skip silently
            logger.debug("Skipping callsite extraction (syntax error) for %s: %s", module.file_path, e)
            return
        except Exception as e:
            logger.debug("Skipping callsite extraction for %s: %s", module.file_path, e)
            return

        functions = list(module.functions) + [m for c in module.classes for m in c.methods]
        
        # Build sets of application-defined names
        app_function_names = {f.name for f in module.functions}
        app_class_names = {c.class_name for c in module.classes}
        app_method_names = {m.name for c in module.classes for m in c.methods}
        app_qualified_names = {f.qualified_name for f in module.functions} | \
                             {c.qualified_name for c in module.classes} | \
                             {m.qualified_name for c in module.classes for m in c.methods}
        
        # Common Python built-in and standard library names
        builtin_names = {
            'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
            'open', 'input', 'type', 'isinstance', 'hasattr', 'getattr', 'setattr',
            'enumerate', 'zip', 'map', 'filter', 'sorted', 'sum', 'min', 'max', 'abs',
            'all', 'any', 'bool', 'bytes', 'chr', 'ord', 'dir', 'eval', 'exec', 'format',
            'frozenset', 'hash', 'help', 'hex', 'id', 'iter', 'next', 'object', 'oct',
            'pow', 'repr', 'reversed', 'round', 'slice', 'super', 'vars'
        }
        
        # Build import map and run FQN inference for this module
        import_map = self._build_import_symbol_map(module)
        fqn_map = self._run_fqn_inference(module.file_path)

        def enclosing_function(line: int) -> PyFunction | None:
            candidates = [f for f in functions if f.start_line <= line <= f.end_line]
            if not candidates:
                return None
            return min(candidates, key=lambda f: f.end_line - f.start_line)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            fn = enclosing_function(node.lineno)
            if not fn:
                continue

            # Extract callee name and receiver
            is_static_call = False
            is_constructor_call = False
            receiver_expr = ""
            receiver_type = None
            
            if isinstance(node.func, ast.Attribute):
                callee_name = node.func.attr
                if isinstance(node.func.value, ast.Name):
                    receiver_expr = node.func.value.id
                    recv_name = receiver_expr

                    # Check if receiver is a class name (static method)
                    if receiver_expr and receiver_expr[0].isupper():
                        is_static_call = True
                        receiver_type = receiver_expr
                    else:
                        # Try FQN inference first - try multiple combinations
                        # The FQN map contains entries like:
                        # 'Greeter.greet': 'test.Greeter.greet'
                        # 'greet': 'test.Greeter.greet'
                        # 'test.Greeter.greet': 'test.Greeter.greet'
                        fqn_result = None

                        # Try various combinations to find the FQN
                        for lookup_key in [
                            f"{recv_name}.{callee_name}",  # e.g., "Greeter.greet"
                            callee_name,  # e.g., "greet"
                            recv_name,  # e.g., "Greeter"
                        ]:
                            fqn_result = fqn_map.get(lookup_key)
                            if fqn_result:
                                break
                        
                        if fqn_result:
                            # Extract just the class name from FQN
                            # e.g., 'test.Greeter.greet' -> 'Greeter'
                            # e.g., '<builtin>.print' -> '<builtin>'
                            parts = fqn_result.rsplit('.', 1)
                            if len(parts) == 2:
                                # Get the part before the method name
                                class_part = parts[0]
                                # Extract just the class name (last component)
                                receiver_type = class_part.split('.')[-1] if '.' in class_part else class_part

                        # Try Scalpel TypeInference
                        if not receiver_type:
                            inferred_recv = self._nearest_type(
                                file_path=module.file_path,
                                function_name=fn.name,
                                variable_name=recv_name,
                                line_number=node.lineno,
                            )
                            if inferred_recv:
                                # Get the first inferred type
                                inferred_type = sorted(inferred_recv)[0]
                                # Extract class name from the type string
                                # Type might be: "Greeter", "test.Greeter", or variable name "greeter"
                                # If it starts with lowercase, it's likely a variable name, not a type
                                if inferred_type and inferred_type[0].isupper():
                                    # Extract just the class name (last component if dotted)
                                    receiver_type = inferred_type.split('.')[-1]
                                # else: skip variable names that start with lowercase

                        # Fallback: import map
                        if not receiver_type:
                            receiver_type = import_map.get(recv_name)
                else:
                    try:
                        receiver_expr = ast.unparse(node.func.value)
                        if len(receiver_expr) > 50:
                            receiver_expr = "<complex_expr>"
                        # Try to extract type from chained constructor calls
                        elif isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Name):
                            receiver_type = node.func.value.func.id
                    except Exception:
                        receiver_expr = "<complex_expr>"
            elif isinstance(node.func, ast.Name):
                callee_name = node.func.id
                receiver_expr = ""
                
                # Try FQN inference for direct function calls
                fqn_result = fqn_map.get(callee_name)
                if fqn_result and fqn_result.startswith('<builtin>.'):
                    # This is a built-in function
                    receiver_type = '<builtin>'
                elif fqn_result:
                    # Extract just the class name from FQN
                    # e.g., 'test.Greeter.greet' -> 'Greeter' (method)
                    # e.g., 'test.hello' -> None (module-level function, no receiver)
                    parts = fqn_result.rsplit('.', 1)
                    if len(parts) == 2:
                        class_part = parts[0]
                        # Check if this is a class method (class_part contains a class name)
                        # If class_part is just a module name (all lowercase), it's a module function
                        if '.' in class_part:
                            # e.g., 'test.Greeter' -> 'Greeter'
                            receiver_type = class_part.split('.')[-1]
                        elif class_part and class_part[0].isupper():
                            # Single component that's a class name
                            receiver_type = class_part
                        # else: module-level function, leave receiver_type as None
                
                # Direct call to a name - check if it's a class (constructor)
                if callee_name and callee_name[0].isupper():
                    is_constructor_call = True
                    is_static_call = True
                    # Try to get FQN for the class
                    if not receiver_type:
                        class_fqn = fqn_map.get(callee_name)
                        if class_fqn:
                            # Extract just the class name
                            receiver_type = class_fqn.split('.')[-1] if '.' in class_fqn else class_fqn
                        else:
                            imported = import_map.get(callee_name)
                            if imported:
                                receiver_type = imported.split('.')[-1] if '.' in imported else imported
            else:
                try:
                    callee_name = ast.unparse(node.func)
                    if len(callee_name) > 50:
                        callee_name = "<complex_call>"
                except Exception:
                    callee_name = "unknown"
                receiver_expr = ""

            arguments: list[PyArgument] = []
            argument_types: list[str | None] = []
            argument_expr: list[str] = []

            # Handle positional arguments
            for arg in node.args:
                try:
                    arg_str = ast.unparse(arg)
                    argument_expr.append(arg_str)

                    if isinstance(arg, ast.Name):
                        arg_name = arg.id
                    elif isinstance(arg, ast.Constant):
                        arg_name = repr(arg.value) if len(repr(arg.value)) < 20 else "<constant>"
                    else:
                        arg_name = "<expr>"

                    arguments.append(PyArgument(name=arg_name))
                    argument_types.append(None)
                except Exception:
                    continue
            
            # Handle keyword arguments
            for keyword in node.keywords:
                try:
                    arg_str = ast.unparse(keyword.value)
                    argument_expr.append(f"{keyword.arg}={arg_str}")
                    
                    # For keyword arguments, use the keyword name
                    arg_name = keyword.arg if keyword.arg else "**kwargs"
                    
                    arguments.append(PyArgument(name=arg_name, annotation=arg_str))
                    argument_types.append(None)
                except Exception:
                    continue

            # Determine if this is a library call or application call
            is_library_call = False
            is_application_call = False
            
            if callee_name in builtin_names:
                is_library_call = True
            elif callee_name in app_function_names or callee_name in app_method_names:
                is_application_call = True
            elif receiver_expr in app_class_names:
                is_application_call = True
            elif is_constructor_call and callee_name in app_class_names:
                is_application_call = True
            else:
                # If not clearly application code, assume it's library
                is_library_call = True

            arg_strs = [a.name for a in arguments]
            method_signature = f"{callee_name}({', '.join(arg_strs)})"
            
            # qualified_module_name will be set from PyModule later

            fn.call_sites.append(
                PyCallSite(
                    method_name=callee_name,
                    method_signature=method_signature,
                    qualified_module_name="",  # Will be set from PyModule later
                    receiver_type=receiver_type,  # Inferred from receiver expression
                    arguments=arguments,
                    argument_types=argument_types,
                    argument_expr=argument_expr,
                    return_type="",  # Type inference not implemented in basic version
                    callee_signature=method_signature,
                    is_static_call=is_static_call,
                    is_constructor_call=is_constructor_call,
                    is_library_call=is_library_call,
                    is_application_call=is_application_call,
                    start_line=node.lineno,
                    start_column=node.col_offset,
                    end_line=node.end_lineno or node.lineno,
                    end_column=node.end_col_offset or node.col_offset,
                )
            )

    def _build_call_graph(self) -> nx.DiGraph:
        """Build a simple call graph from the symbol table."""
        if self._app is None:
            return nx.DiGraph()

        g = nx.DiGraph()

        # Build resolution maps
        name_to_qualified: Dict[str, List[str]] = {}
        for mod in self._app.symbol_table.values():
            for f in mod.functions:
                name_to_qualified.setdefault(f.name, []).append(f.qualified_name)
            for c in mod.classes:
                for mtd in c.methods:
                    name_to_qualified.setdefault(mtd.name, []).append(mtd.qualified_name)

        def _resolve_callee(raw: str) -> str:
            if raw in name_to_qualified and len(set(name_to_qualified[raw])) == 1:
                return list(set(name_to_qualified[raw]))[0]
            return raw

        # Create graph based on callsites
        for mod in self._app.symbol_table.values():
            for f in mod.functions:
                for cs in f.call_sites:
                    g.add_edge(f.qualified_name, _resolve_callee(cs.method_name))
            for c in mod.classes:
                for mtd in c.methods:
                    for cs in mtd.call_sites:
                        g.add_edge(mtd.qualified_name, _resolve_callee(cs.method_name))

        return g

    def get_application_view(self) -> PyApplication:
        if self._app is not None:
            return self._app

        cached = self._load_cached()
        if cached is not None:
            self._app = cached
            return cached

        modules: Dict[str, PyModule] = {}

        if self.project_dir:
            all_files = list(self.project_dir.rglob("*.py"))

            # Step 1: Run type inference on all modules and index results
            desc = "Running type inference"
            it = tqdm(all_files, desc=desc, disable=not self.show_progress) if self.show_progress else all_files
            
            for file_path in it:
                try:
                    type_records = self._run_scalpel_type_inference(file_path)
                    for record in type_records:
                        self._index_inferred_item(record, str(file_path.resolve()))
                except Exception:
                    # All errors are already handled in _run_scalpel_type_inference
                    # No need to log again here
                    pass

            # Step 2: Parse modules and attach callsites with type information
            desc = "Parsing modules"
            it = tqdm(all_files, desc=desc, disable=not self.show_progress) if self.show_progress else all_files

            for file_path in it:
                try:
                    mod = self._parse_module(file_path)
                    self._attach_callsites(mod)
                    
                    # Set qualified_module_name from the module for all functions and methods
                    for func in mod.functions:
                        func.qualified_module_name = mod.qualified_name
                        for cs in func.call_sites:
                            cs.qualified_module_name = mod.qualified_name
                    
                    for cls in mod.classes:
                        for method in cls.methods:
                            method.qualified_module_name = mod.qualified_name
                            for cs in method.call_sites:
                                cs.qualified_module_name = mod.qualified_name
                    
                    modules[str(file_path)] = mod
                except Exception as e:
                    # Log parse errors at debug level - these are expected for some files
                    logger.debug("Skipping file (parse error) %s: %s", file_path, str(e)[:100])

        self._app = PyApplication(symbol_table=modules)
        self._persist(self._app)

        return self._app

    def get_symbol_table(self) -> Dict[str, PyModule]:
        return self.get_application_view().symbol_table

    def get_call_graph(self) -> nx.DiGraph:
        if self._call_graph is None:
            self._call_graph = self._build_call_graph()
        return self._call_graph