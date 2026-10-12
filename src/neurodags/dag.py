# dag_exec.py
import inspect
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    import xarray as xr
except ImportError:  # pragma: no cover - optional dependency
    xr = None

from neurodags.definitions import NodeResult, SkipDerivative
from neurodags.derivatives import (
    get_derivative as get_derivative_definition,
)
from neurodags.derivatives import (
    list_derivatives as list_derivative_definitions,
)
from neurodags.derivatives import (
    register_derivative_with_name,
)
from neurodags.loaders import load_configuration
from neurodags.loggers import get_logger
from neurodags.naming import (
    ColumnCollisionError,
    find_column_collisions,
    format_column,
    validate_artifact_suffixes,
)
from neurodags.nodes import get_node
from neurodags.outputs import (
    build_manifest,
    manifest_artifacts,
    output_prefix,
    publish_marker,
    remove_markers,
    unit_state,
    write_manifest,
    writer_paths,
)

log = get_logger(__name__)
# A node reference is ``id.N`` (whole result of node N) or, for a node that
# returns a mapping (a "parser node"), ``id.N.field`` to pull a single field.
# The field must be a plain identifier: ``.`` is the reference separator, so
# dotted/nested keys would be ambiguous and are rejected (see _resolve_refs).
_ID_REF = re.compile(r"^id\.(\d+)(?:\.([A-Za-z_]\w*))?$")
# Anything shaped ``id.<digits>.<rest>`` that is NOT the single-field form above
# (nested paths, dashed keys, trailing dot, ...) — used to raise a clear error
# instead of silently passing the string through as a literal.
_ID_REF_PREFIX = re.compile(r"^id\.\d+\.")
_VAR_REF = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*)$")

# ---------- helpers ----------


def _is_id_ref(x: Any) -> bool:
    return isinstance(x, str) and _ID_REF.match(x) is not None


def _node_result_to_mapping(val: Any, ref: str) -> Any:
    """Best-effort conversion of a stored node value to the mapping it carries.

    Parser-style nodes return a ``NodeResult`` wrapping a single dict artifact;
    ``id.N.field`` access needs that dict. When the referenced derivative was
    served from cache, the store holds the artifact's *file path* instead, so a
    ``.json`` artifact on disk is loaded back into a mapping. Any other existing
    artifact file raises a clear error (only ``.json`` can be field-accessed).
    Returns ``val`` unchanged when it is already a mapping or cannot be
    converted to one (the caller then raises).
    """
    if isinstance(val, NodeResult):
        if len(val.artifacts) == 1:
            return next(iter(val.artifacts.values())).item
        for art in val.artifacts.values():
            if isinstance(art.item, Mapping):
                return art.item
        return val
    if isinstance(val, str | os.PathLike):
        path = Path(val)
        if path.is_file():
            if path.suffix.lower() != ".json":
                raise ValueError(
                    f"Field access '{ref}' resolved to the file '{path.as_posix()}' "
                    f"(e.g. a cached derivative artifact); only '.json' artifacts can be "
                    f"loaded for field access. Reference the parser derivative's JSON "
                    f"artifact explicitly, e.g. 'derivative: <Name>.json'."
                )
            try:
                with open(path, encoding="utf-8") as handle:
                    return json.load(handle)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Field access '{ref}' could not load JSON artifact "
                    f"'{path.as_posix()}': {exc}"
                ) from exc
    return val


def _resolve_refs(obj: Any, store: dict[int, Any]) -> Any:
    if isinstance(obj, str):
        m = _ID_REF.match(obj)
        if m:
            sid = int(m.group(1))
            field = m.group(2)
            if sid not in store:
                raise KeyError(f"Reference id.{sid} not computed yet")
            if field is None:
                return store[sid]
            container = _node_result_to_mapping(store[sid], obj)
            if not isinstance(container, Mapping):
                raise ValueError(
                    f"Field access '{obj}' requires node {sid} to return a mapping "
                    f"(dict); got {type(container).__name__}."
                )
            if field not in container:
                raise KeyError(
                    f"Field '{field}' not found in id.{sid}; "
                    f"available fields: {sorted(container)}"
                )
            return container[field]
        if _ID_REF_PREFIX.match(obj):
            raise ValueError(
                f"Unsupported reference '{obj}': use 'id.N' or single-level "
                f"'id.N.field' where field is a plain identifier "
                f"(letters/digits/underscore). Nested, dotted, or dashed field "
                f"paths are not supported."
            )
        return obj
    if isinstance(obj, dict):
        return {k: _resolve_refs(v, store) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        t = type(obj)
        return t(_resolve_refs(v, store) for v in obj)
    return obj


def _resolve_vars(obj: Any, vars_dict: dict[str, Any]) -> Any:
    """Substitute $var_name references using dataset-level vars.

    Only whole-string values matching ``$identifier`` are substituted.
    Partial matches (e.g. embedded ``$`` in a path) are left untouched.
    NodeResult values are passed through unchanged (they are runtime objects,
    not YAML strings, and must not be iterated as tuples).
    """
    if isinstance(obj, NodeResult):
        return obj
    if isinstance(obj, str):
        m = _VAR_REF.match(obj)
        if m:
            key = m.group(1)
            if key not in vars_dict:
                raise KeyError(
                    f"Dataset var '${key}' referenced in pipeline args but not defined "
                    f"in the active dataset entry's 'vars:' block. "
                    f"Available vars: {sorted(vars_dict)}"
                )
            return vars_dict[key]
        return obj
    if isinstance(obj, dict):
        return {k: _resolve_vars(v, vars_dict) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        t = type(obj)
        return t(_resolve_vars(v, vars_dict) for v in obj)
    return obj


def _collect_id_refs(obj: Any) -> set[int]:
    """Find every `id.<n>` string embedded in the given object."""
    refs: set[int] = set()
    if isinstance(obj, str):
        match = _ID_REF.match(obj)
        if match:
            refs.add(int(match.group(1)))
        return refs
    if isinstance(obj, dict):
        for value in obj.values():
            refs.update(_collect_id_refs(value))
        return refs
    if isinstance(obj, list | tuple | set):
        for value in obj:
            refs.update(_collect_id_refs(value))
        return refs
    return refs


def _unwrap_for_arg(val: Any, argname: str) -> Any:
    """Heuristic: unwrap NodeResult when passing to a primitive function."""
    if isinstance(val, NodeResult):
        if argname in {"data", "dataset", "da", "ds"} and hasattr(val, "data"):
            return val.data
        if argname in {"mne_object", "meeg", "raw", "epochs"}:
            # Prefer a .fif artifact if present
            for k, art in val.artifacts.items():
                if k == ".fif" or k.endswith(".fif"):
                    return art.item
        # Single-artifact unwrap
        if len(val.artifacts) == 1:
            return next(iter(val.artifacts.values())).item
    return val


def _prep_kwargs(
    raw_kwargs: dict[str, Any],
    store: dict[int, Any],
    vars_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_refs(raw_kwargs or {}, store)
    if vars_dict is not None:
        resolved = _resolve_vars(resolved, vars_dict)
    return {k: _unwrap_for_arg(v, k) for k, v in resolved.items()}


def _topo_order(nodes: list[dict[str, Any]]) -> list[int]:
    steps = {s["id"]: s for s in nodes}
    dependencies: dict[int, set[int]] = {}
    for sid, step in steps.items():
        declared = step.get("depends_on") or []
        deps = set(declared)
        if "args" in step:
            deps.update(_collect_id_refs(step.get("args")))
        deps.discard(sid)
        dependencies[sid] = deps
    seen, order = set(), []

    def visit(i: int):
        if i in seen:
            return
        for d in sorted(dependencies.get(i, set())):
            visit(d)
        seen.add(i)
        order.append(i)

    for i in sorted(steps):
        visit(i)
    return order


# --- registration from YAML ---
def register_derivatives_from_yaml(yaml_path: str) -> list[str]:
    cfg = load_configuration(yaml_path)
    derivative_defs: dict[str, dict] = cfg.get("DerivativeDefinitions", {}) or {}
    registered: list[str] = []

    for derivative_name, derivative_def in derivative_defs.items():

        def make_wrapper(derivative_name: str, derivative_def: dict):
            def wrapper(
                file_path: str,
                reference_base: Path | None = None,
                dataset_config: Any = None,
                mount_point: Path | None = None,
            ) -> NodeResult:
                return run_derivative(
                    derivative_def,
                    file_path,
                    reference_base or Path(""),
                    dataset_config=dataset_config,
                    mount_point=mount_point,
                )

            return wrapper

        func = make_wrapper(derivative_name, derivative_def)

        # pass the definition into the registry
        register_derivative_with_name(derivative_name, func, definition=derivative_def)

        registered.append(derivative_name)
        log.debug("Registered derivative", name=derivative_name)

    return registered


def _ensure_parent_dir(reference_base: str) -> None:
    parent = os.path.dirname(reference_base)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _cached_unit(prefix: str, name: str) -> dict[str, list[str]] | None:
    """Files per suffix of a done unit (valid ``.done`` manifest), else ``None`` (spec 6.2)."""
    return manifest_artifacts(prefix, name)


def _main_files(artifacts: Mapping[str, Sequence[str]]) -> list[str]:
    """The main file of every artifact (first listed path), sorted like the 0.3.x glob.

    MNE split parts follow their main file in the manifest and are loaded through it, so
    they are never handed to nodes on their own.
    """
    return sorted(paths[0] for paths in artifacts.values() if paths)


class _MissingPrecomputedArtifacts(RuntimeError):
    """Raised when dataframe-only collection requires cached artifacts but none are available."""


def _split_derivative_ref(derivative_ref: str) -> tuple[str, str | None]:
    if "." in derivative_ref:
        base, ext = derivative_ref.split(".", 1)
        return base, ext
    return derivative_ref, None


# ---------- executor ----------


def run_derivative(
    derivative_def: dict,
    derivative_name: str,
    file_path: str,
    reference_base: Path,
    dataset_config: Any = None,
    mount_point: Path | None = None,
    dry_run: bool = False,
    skip_errors: bool = False,
    *,
    source_dataset: str | None = None,
    source_rel: str | None = None,
) -> NodeResult | dict[str, Any]:
    """
    Evaluate (or dry-run) a derivative on a single file.

    Outputs of the unit are ``<reference_base>@<derivative_name><suffix>``, with the
    derivative name used verbatim (see :mod:`neurodags.outputs`). A unit is cached when its
    ``.done`` manifest is valid (spec 6.2); files without one are recomputed.

    When dry_run=True:
      - Do NOT execute any steps.
      - Report, per step, whether the expected artifacts are already present (``state``
        of the final output: done, error, skipped, incomplete or missing).
      - Return a dict with a 'plan' list (so callers don't confuse it with NodeResult).

    Normal mode:
      - 'derivative' steps: if cached and overwrite=False, skip compute; else compute or recurse.
      - 'node' steps: execute; if it's the last step, validate the artifact suffixes, delete
        the old ``.done``, write every artifact, then publish the ``.done`` manifest
        atomically from the paths the writers reported.

    Parameters
    ----------
    source_dataset, source_rel : str, optional
        Dataset key and source path relative to ``source_root``, recorded in the
        manifest's ``source``. ``source_rel`` defaults to the source file's basename.
    """
    overwrite = bool(derivative_def.get("overwrite", False))
    save = bool(derivative_def.get("save", True))
    nodes = derivative_def.get("nodes", [])
    order = _topo_order(nodes)
    manifest_rel = source_rel if source_rel is not None else os.path.basename(file_path)

    store: dict[int, Any] = {}
    last_result: NodeResult | None = None
    plan: list[dict[str, Any]] = [] if dry_run else None

    def _record(**kwargs):
        if plan is not None:
            plan.append(kwargs)

    # In dry-run, also check the final output of the derivative upfront
    base_reference = os.fspath(reference_base)
    final_prefix = output_prefix(base_reference, derivative_name) if save else None
    error_path = (final_prefix + ".error") if final_prefix else None
    skip_marker_path = (final_prefix + ".skip") if final_prefix else None
    has_error_marker = bool(error_path and Path(error_path).exists())
    has_skip_marker = bool(skip_marker_path and Path(skip_marker_path).exists())

    if dry_run:
        if save:
            state = unit_state(final_prefix, derivative_name)
            _record(
                id="final",
                kind="derivative_output",
                name=derivative_name,
                prefix=final_prefix,
                state=state.state,
                cached=state.state == "done",
                paths=_main_files(state.artifacts),
                artifacts=state.artifacts,
                origin=state.origin,
                manifest_problem=state.problem,
                unlisted=list(state.unlisted),
                has_error_marker=has_error_marker,
                error_path=error_path if has_error_marker else None,
                has_skip_marker=has_skip_marker,
                skip_path=skip_marker_path if has_skip_marker else None,
            )
        else:
            _record(
                id="final",
                kind="derivative_output",
                name=derivative_name,
                will_save=False,
                cached=False,
                paths=[],
                has_error_marker=has_error_marker,
                error_path=error_path if has_error_marker else None,
                has_skip_marker=has_skip_marker,
                skip_path=skip_marker_path if has_skip_marker else None,
            )
    else:
        # Early skip if .skip marker exists (derivative not applicable), unless overwriting
        if has_skip_marker and not overwrite:
            log.info(
                "Derivative has skip marker; treating as not applicable",
                derivative=derivative_name,
                skip_path=skip_marker_path,
                file_path=file_path,
            )
            return {"skipped": skip_marker_path}
        # Early skip if the unit is done (valid .done manifest) and not overwriting
        if save and not overwrite:
            cached = _cached_unit(final_prefix, derivative_name)
            if cached is not None:
                final_candidates = _main_files(cached)
                log.info(
                    "Last step of the derivative is cached",
                    derivative=derivative_name,
                    final_prefix=final_prefix,
                    reference_base=reference_base,
                    candidate=final_candidates[0] if final_candidates else None,
                )
                return {"cached": final_candidates, "artifacts": cached}
        # Early skip if prior error marker exists and skip_errors=True
        if skip_errors and has_error_marker:
            log.info(
                "Skipping file with prior error marker",
                derivative=derivative_name,
                error_path=error_path,
                file_path=file_path,
            )
            return {"skipped_error": error_path}
        # About to compute: make sure the output directory exists for artifact
        # writers, .error/.skip markers and nodes writing next to reference_base.
        # Done here (not when resolving reference_base) so dry-runs and cached
        # dataframe reads never touch the filesystem (#33).
        _ensure_parent_dir(base_reference)

    def _write_skip(text: str) -> str:
        assert final_prefix is not None
        path = publish_marker(final_prefix, "skip", text)
        log.debug("Wrote skip marker", path=path)
        return path

    for sid in order:
        step = next(s for s in nodes if s["id"] == sid)

        # 1) 'derivative' step
        if "derivative" in step:
            derivative_ref = step["derivative"]

            if derivative_ref == "SourceFile":
                store[sid] = file_path
                _record(id=sid, kind="source", name="SourceFile", path=file_path)
                continue

            base_name, ext = _split_derivative_ref(derivative_ref)
            prefix = output_prefix(base_reference, base_name)
            child_cached = _cached_unit(prefix, base_name)
            candidates = _main_files(child_cached) if child_cached is not None else []
            if ext:
                candidates = [p for p in candidates if p.endswith("." + ext)]

            if dry_run:
                entry: dict[str, Any] = {
                    "id": sid,
                    "kind": "derivative",
                    "name": derivative_ref,
                    "prefix": prefix,
                    "cached": len(candidates) > 0,
                    "paths": candidates,
                }
                if base_name in list_derivative_definitions():
                    sub_def = get_derivative_definition(base_name).definition
                    entry["subderivative_plan"] = run_derivative(
                        sub_def,
                        base_name,
                        file_path,
                        reference_base,
                        dataset_config=dataset_config,
                        mount_point=mount_point,
                        dry_run=True,
                        source_dataset=source_dataset,
                        source_rel=source_rel,
                    )
                _record(**entry)
                continue

            # normal execution path
            # Use the child derivative's own overwrite flag, not the parent's.
            # A parent with overwrite=True should recompute its own node, but
            # should still respect a child's overwrite=False (i.e. use its cache).
            child_overwrite = overwrite  # fallback: honour parent flag for unknown children
            if base_name in list_derivative_definitions():
                child_overwrite = bool(
                    get_derivative_definition(base_name).definition.get("overwrite", False)
                )
            cached_here = (len(candidates) > 0) and not child_overwrite
            if cached_here:
                log.debug(
                    "Using cached derivative",
                    derivative=derivative_name,
                    id=sid,
                    child_derivative=derivative_ref,
                    paths=candidates,
                )
                store[sid] = candidates[0]  # marker
                if len(candidates) > 1:
                    log.warning(
                        "Multiple cached artifacts found, using the first one",
                        derivative=derivative_name,
                        child_derivative=derivative_ref,
                        reference_base=reference_base,
                        candidates=candidates,
                        selected=candidates[0],
                    )
                last_result = {"cached": candidates}
                continue

            if base_name in list_derivative_definitions():
                log.debug(
                    "Recurse into sub-derivative",
                    parent_derivative=derivative_name,
                    child_derivative=base_name,
                )
                try:
                    sub_result = run_derivative(
                        get_derivative_definition(base_name).definition,
                        base_name,
                        file_path,
                        reference_base=reference_base,
                        dataset_config=dataset_config,
                        mount_point=mount_point,
                        dry_run=False,
                        source_dataset=source_dataset,
                        source_rel=source_rel,
                    )
                except SkipDerivative as skip_exc:
                    # save:False sub-derivative raised SkipDerivative — propagate skip upward
                    log.info(
                        "Sub-derivative (save=False) signaled skip; propagating to parent",
                        parent_derivative=derivative_name,
                        child_derivative=base_name,
                        reason=str(skip_exc),
                    )
                    if save:
                        return {
                            "skipped": _write_skip(
                                f"Sub-derivative '{base_name}' was skipped: {skip_exc!s}\n"
                            )
                        }
                    raise  # save:False parent also propagates

                # If sub-derivative returned {"skipped": ...} (save:True sub wrote its own
                # .skip marker), propagate skip to this parent derivative too.
                if isinstance(sub_result, dict) and "skipped" in sub_result:
                    log.info(
                        "Sub-derivative was skipped; propagating skip to parent",
                        parent_derivative=derivative_name,
                        child_derivative=base_name,
                    )
                    if save:
                        return {
                            "skipped": _write_skip(f"Sub-derivative '{base_name}' was skipped.\n")
                        }
                    raise SkipDerivative(f"Sub-derivative '{base_name}' was skipped")

                # If the sub-derivative early-returned its cached-file dict
                # ({"cached": [path, ...]}) instead of a NodeResult, resolve it
                # to the matching path so downstream nodes receive a plain path
                # string rather than the internal bookkeeping dict.
                if isinstance(sub_result, dict) and "cached" in sub_result:
                    cached_paths = sub_result["cached"]
                    if ext:
                        filtered = [p for p in cached_paths if p.endswith("." + ext)]
                        store[sid] = (
                            filtered[0]
                            if filtered
                            else (cached_paths[0] if cached_paths else sub_result)
                        )
                    else:
                        store[sid] = cached_paths[0] if cached_paths else sub_result
                    last_result = last_result  # cached sub-derivative; last_result unchanged
                    continue

                # If a specific artifact suffix was requested (e.g. "derivative:
                # Splitter.condA.fif"), narrow the in-memory NodeResult to just
                # that artifact so downstream nodes receive the right object.
                # This mirrors the on-disk filter already applied via
                # `candidates = [p for p in candidates if p.endswith("." + ext)]`.
                if isinstance(sub_result, NodeResult) and ext:
                    target = "." + ext
                    if target in sub_result.artifacts:
                        sub_result = NodeResult(artifacts={target: sub_result.artifacts[target]})
                    else:
                        log.warning(
                            "Artifact suffix not found in sub-derivative; "
                            "passing full NodeResult",
                            requested_suffix=target,
                            available=list(sub_result.artifacts.keys()),
                            parent_derivative=derivative_name,
                            child_derivative=derivative_ref,
                        )
                store[sid] = sub_result
                last_result = sub_result if isinstance(sub_result, NodeResult) else last_result
            else:
                log.error(
                    "Unknown derivative reference",
                    derivative=derivative_name,
                    id=sid,
                    child_derivative=derivative_ref,
                )
                raise ValueError(
                    f"Unknown derivative '{derivative_ref}' in derivative '{derivative_name}'"
                )

        # 2) 'node' step
        elif "node" in step:
            node_name = step["node"]

            if dry_run:
                if not sid == order[-1]:
                    _record(
                        id=sid,
                        kind="node",
                        name=node_name,
                        will_compute=True,
                        is_final=(sid == order[-1]),
                    )
                continue
                # not sure what to do if not

            fn = get_node(node_name)

            extra_args = {}
            if "reference_base" in inspect.signature(fn).parameters:
                extra_args["reference_base"] = Path(reference_base)
            if "dataset_config" in inspect.signature(fn).parameters:
                extra_args["dataset_config"] = dataset_config
            if "mount_point" in inspect.signature(fn).parameters:
                extra_args["mount_point"] = mount_point

            _vars = dataset_config.vars if dataset_config is not None else None
            kwargs = _prep_kwargs(step.get("args", {}), store, vars_dict=_vars)
            log.debug(
                "Execute node", derivative=derivative_name, id=sid, node=node_name, kwargs=kwargs
            )
            try:
                res = fn(**kwargs, **extra_args)
                if not isinstance(res, NodeResult):
                    raise TypeError(f"Node {node_name} must return NodeResult")
                store[sid] = res
                last_result = res

                if sid == order[-1] and save:
                    assert final_prefix is not None
                    _save_final_artifacts(
                        res,
                        final_prefix,
                        derivative_name,
                        source_dataset=source_dataset,
                        source_rel=manifest_rel,
                    )
            except SkipDerivative as skip_exc:
                log.info(
                    "Node signaled skip; derivative not applicable for this source file",
                    derivative=derivative_name,
                    id=sid,
                    node=node_name,
                    reason=str(skip_exc),
                )
                if save:
                    try:
                        path = _write_skip(
                            f"Derivative '{derivative_name}' step id={sid} node='{node_name}' skipped:\n{skip_exc!s}\n"
                        )
                    except Exception as ee:
                        log.error(
                            "Error writing skip marker file",
                            path=skip_marker_path,
                            error=str(ee),
                            exc_info=True,
                        )
                        path = skip_marker_path
                    return {"skipped": path}
                raise  # save:False — propagate to parent
            except Exception as e:
                log.error(
                    "Error executing node",
                    derivative=derivative_name,
                    id=sid,
                    node=node_name,
                    error=str(e),
                    exc_info=True,
                )
                # try to save a dummy file .error to mark failure
                if save:
                    try:
                        assert final_prefix is not None
                        error_path = publish_marker(
                            final_prefix,
                            "error",
                            f"Derivative '{derivative_name}' step id={sid} node='{node_name}' failed:\n{e!s}\n",
                        )
                        log.debug("Wrote error marker file", path=error_path)
                    except Exception as ee:
                        log.error(
                            "Error writing error marker file",
                            path=final_prefix + ".error" if final_prefix else None,
                            error=str(ee),
                            exc_info=True,
                        )
                raise
        else:
            raise ValueError(f"Step id={sid} must specify 'derivative' or 'node'")

    if dry_run:
        return {
            "derivative": derivative_name,
            "file": file_path,
            "reference_base": base_reference,
            "overwrite": overwrite,
            "plan": plan,
        }

    if last_result is None:
        raise RuntimeError(f"Derivative '{derivative_name}' produced no result")
    return last_result


def _save_final_artifacts(
    res: NodeResult,
    final_prefix: str,
    derivative_name: str,
    *,
    source_dataset: str | None,
    source_rel: str,
) -> str:
    """Write a unit's artifacts and publish its ``.done`` manifest (spec 3, 6.2).

    1. Validate every artifact suffix before the first writer runs (an invalid key fails
       the unit and nothing is written).
    2. Delete the old ``.done``, so a crash from here on leaves the unit unfinished.
    3. Run every writer and record the files it reports (:func:`neurodags.outputs.writer_paths`).
    4. Write ``.done`` atomically, then remove stale ``.error``/``.skip`` markers.

    Returns the manifest path. Errors propagate (the caller writes the ``.error`` marker).
    """
    validate_artifact_suffixes(res.artifacts)
    remove_markers(final_prefix, "done")
    written: dict[str, list[str]] = {}
    for artifact_name, artifact in res.artifacts.items():
        artifact_path = final_prefix + artifact_name
        try:
            log.debug("Processed artifact", name=artifact_name, file=artifact)
            written[artifact_name] = writer_paths(artifact_path, artifact.writer(artifact_path))
            log.debug("Saved artifact", file=artifact_path, files=written[artifact_name])
        except Exception as e:
            log.error(
                "Error saving artifact",
                artifact_name=artifact_name,
                path=artifact_path,
                error=str(e),
                exc_info=True,
            )
            raise
    manifest = build_manifest(
        final_prefix,
        derivative_name,
        dataset=source_dataset,
        rel=source_rel,
        artifacts=written,
    )
    manifest_path = write_manifest(final_prefix, manifest)
    remove_markers(final_prefix, "error", "skip")
    log.debug("Published completion manifest", path=manifest_path)
    return manifest_path


def collect_derivative_for_dataframe(
    derivative_def: dict,
    derivative_name: str,
    file_path: str,
    reference_base: Path,
    dataset_config: Any = None,
    mount_point: Path | None = None,
    *,
    flatten_xarray_artifacts: bool = True,
    preserve_complex_values: bool = False,
    source_dataset: str | None = None,
    source_rel: str | None = None,
) -> dict[str, Any]:
    """
    Collect artifacts for a derivative and convert them into dataframe-ready values.

    This helper prefers cached artifacts: a unit with a valid ``.done`` manifest is read
    from the files the manifest lists. Otherwise it executes ``run_derivative`` to obtain
    the results (without forcing a recompute when cached outputs are already present).

    Parameters
    ----------
    flatten_xarray_artifacts:
        When True, ``xr.DataArray`` artifacts (or paths to ``.nc`` files containing them)
        are flattened into multiple columns using their coordinate metadata.

    preserve_complex_values:
        When True, skip flattening and value simplification so artifacts retain their original
        Python objects (after potential on-disk loading).

    source_dataset, source_rel:
        Recorded in the ``.done`` manifest when the derivative has to be computed (see
        :func:`run_derivative`).

    Returns
    -------
    dict[str, Any]
        A dictionary suitable for composing a DataFrame row, with columns named by
        :func:`neurodags.naming.format_column`: ``{derivative_name}{artifact_suffix}``, plus
        ``@dim-value_dim-value`` (percent-encoded, dims sorted by encoded name) per flattened
        element. Derivatives flagged with ``save=True`` and ``for_dataframe=True`` are skipped
        (empty dictionary) when the unit has no valid ``.done`` manifest instead of
        triggering a recomputation.

    Raises
    ------
    neurodags.naming.ColumnCollisionError
        When two coordinates or artifacts produce the same column.
    """

    enforce_precomputed = bool(derivative_def.get("save", True)) and bool(
        derivative_def.get("for_dataframe", False)
    )

    try:
        artifact_payloads = _resolve_derivative_artifacts(
            derivative_def,
            derivative_name,
            file_path,
            reference_base,
            dataset_config=dataset_config,
            mount_point=mount_point,
            precomputed_only=enforce_precomputed,
            source_dataset=source_dataset,
            source_rel=source_rel,
        )
    except _MissingPrecomputedArtifacts:
        log.debug(
            "Skipping derivative during dataframe collection because cached artifacts are missing",
            derivative=derivative_name,
            file=file_path,
            reference_base=os.fspath(reference_base),
        )
        return {}

    row_values: dict[str, Any] = {}
    # (origin, column) for every generated column, so collisions name both origins (spec 7.1).
    origins: list[tuple[str, str]] = []
    for suffix, payload in artifact_payloads.items():
        simplified = _simplify_artifact_payload(
            payload,
            suffix=suffix,
            flatten_xarray=flatten_xarray_artifacts and not preserve_complex_values,
            preserve_complex_values=preserve_complex_values,
            derivative_name=derivative_name,
        )
        if isinstance(simplified, _FlattenedArtifactColumns):
            for target_column, value in simplified.items():
                origins.append((simplified.origins[target_column], target_column))
                row_values[target_column] = value
        else:
            column_name = format_column(derivative_name, suffix)
            origins.append((f"artifact {derivative_name}{suffix}", column_name))
            row_values[column_name] = simplified
    _raise_on_column_collisions(origins)
    return row_values


def _resolve_derivative_artifacts(
    derivative_def: dict,
    derivative_name: str,
    file_path: str,
    reference_base: Path,
    dataset_config: Any = None,
    mount_point: Path | None = None,
    *,
    precomputed_only: bool = False,
    source_dataset: str | None = None,
    source_rel: str | None = None,
) -> dict[str, Any]:
    save = bool(derivative_def.get("save", True))
    prefix = output_prefix(os.fspath(reference_base), derivative_name) if save else None

    def _from_manifest(artifacts: Mapping[str, Sequence[str]]) -> dict[str, Any]:
        # The main file of each artifact (MNE split parts are read through it).
        return {suffix: Path(paths[0]) for suffix, paths in artifacts.items() if paths}

    if save:
        cached = _cached_unit(prefix, derivative_name)
        if cached is not None:
            return _from_manifest(cached)
        if precomputed_only:
            raise _MissingPrecomputedArtifacts(
                f"No valid .done manifest for derivative '{derivative_name}' "
                f"at prefix '{prefix}'"
            )

    result = run_derivative(
        derivative_def,
        derivative_name,
        file_path,
        reference_base,
        dataset_config=dataset_config,
        mount_point=mount_point,
        dry_run=False,
        source_dataset=source_dataset,
        source_rel=source_rel,
    )

    if isinstance(result, NodeResult):
        written = _cached_unit(prefix, derivative_name) if save else None
        payloads: dict[str, Any] = {}
        for suffix, artifact in result.artifacts.items():
            payload: Any = artifact.item
            if written and written.get(suffix):
                payload = Path(written[suffix][0])
            payloads[suffix] = payload
        return payloads

    if isinstance(result, dict) and "artifacts" in result:
        return _from_manifest(result["artifacts"])

    raise RuntimeError(
        f"Unexpected result type when collecting derivative '{derivative_name}': {type(result)!r}"
    )


def _simplify_artifact_payload(
    payload: Any,
    *,
    suffix: str,
    flatten_xarray: bool = False,
    preserve_complex_values: bool = False,
    derivative_name: str,
) -> Any:
    value = _load_from_path(payload, suffix=suffix) if isinstance(payload, Path) else payload

    if preserve_complex_values:
        return value

    if flatten_xarray and xr is not None and isinstance(value, xr.DataArray):
        return _flatten_dataarray_payload(value, derivative_name=derivative_name, suffix=suffix)

    return _simplify_value(value)


def _load_from_path(path: Path, *, suffix: str) -> Any:
    try:
        if suffix.endswith(".json"):
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        if suffix.endswith(".nc") and xr is not None:
            try:
                data_array = xr.open_dataarray(path)
                loaded = data_array.load()
                data_array.close()
                return loaded
            except Exception:
                dataset = xr.open_dataset(path)
                loaded = dataset.load()
                dataset.close()
                return loaded
        if suffix.endswith(".txt") or suffix.endswith(".log"):
            return path.read_text(encoding="utf-8")
    except Exception:
        log.warning(
            "Failed to load artifact payload, falling back to path",
            path=str(path),
            suffix=suffix,
            exc_info=True,
        )
    return path.as_posix()


class _FlattenedArtifactColumns(dict):
    """Marker mapping ``{column: value}`` for a DataArray expanded into several columns.

    ``origins`` maps each column to a description of the coordinates behind it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.origins: dict[str, str] = {}


def _raise_on_column_collisions(origins: Sequence[tuple[str, str]]) -> None:
    """Raise :class:`ColumnCollisionError` when two origins produce the same column."""
    collisions = find_column_collisions(origins)
    if not collisions:
        return
    shown = list(collisions.items())[:5]
    details = "; ".join(f"{column!r} from " + " and ".join(found) for column, found in shown)
    more = f" (and {len(collisions) - len(shown)} more)" if len(collisions) > len(shown) else ""
    raise ColumnCollisionError(
        f"Dataframe column collision: {details}{more}. Each column must come from exactly one "
        "coordinate of one artifact; make the coordinate index unique and use one type per "
        "coordinate (e.g. not both 1 and '1')."
    )


def _coord_label(value: Any) -> Any:
    """Unwrap numpy scalars so the column shows ``1.5``, not ``np.float64(1.5)``."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _flatten_dataarray_payload(
    data_array: "xr.DataArray",
    *,
    derivative_name: str,
    suffix: str,
) -> _FlattenedArtifactColumns:
    """Expand a DataArray into ``{column: value}`` with :func:`naming.format_column` columns.

    Raises
    ------
    ColumnCollisionError
        When two elements map to the same column (non-unique index, or mixed types such
        as ``1`` and ``"1"``).
    """
    artifact = f"{derivative_name}{suffix}"
    if data_array.ndim == 0:
        try:
            scalar = data_array.item()
        except Exception:
            scalar = data_array.values
        column = format_column(derivative_name, suffix)
        flattened = _FlattenedArtifactColumns({column: _simplify_value(scalar)})
        flattened.origins[column] = f"artifact {artifact}"
        return flattened

    dims = list(data_array.dims)
    shape = tuple(data_array.sizes[dim] for dim in dims)
    index_iterator = _ndindex(shape)

    flattened = _FlattenedArtifactColumns()
    origins: list[tuple[str, str]] = []
    for index_tuple in index_iterator:
        dims_with_values = []
        for dim, idx in zip(dims, index_tuple, strict=False):
            coord_value = None
            if dim in data_array.indexes:
                try:
                    coord_value = data_array.indexes[dim][idx]
                except Exception:
                    coord_value = idx
            elif dim in data_array.coords:
                coord = data_array.coords[dim]
                try:
                    coord_value = coord.values[idx]
                except Exception:
                    coord_value = coord[idx]
            else:
                coord_value = idx
            dims_with_values.append((dim, _coord_label(coord_value)))

        # Dims are percent-encoded and sorted by encoded name inside format_column.
        column = format_column(derivative_name, suffix, dims_with_values)

        try:
            raw_value = data_array.data[index_tuple]
        except Exception:
            raw_value = data_array.values[index_tuple]

        origin = f"artifact {artifact} at " + ", ".join(
            f"{dim}={value!r}" for dim, value in dims_with_values
        )
        origins.append((origin, column))
        flattened.origins[column] = origin
        flattened[column] = _simplify_value(raw_value)

    _raise_on_column_collisions(origins)
    return flattened


def _ndindex(shape: Sequence[int]):
    if np is not None:
        return np.ndindex(*shape)
    return _ndindex_fallback(shape)


def _ndindex_fallback(shape: Sequence[int]):
    if not shape:
        yield ()
        return
    indices = [0] * len(shape)
    while True:
        yield tuple(indices)
        for axis in reversed(range(len(shape))):
            indices[axis] += 1
            if indices[axis] < shape[axis]:
                break
            indices[axis] = 0
            if axis == 0:
                return


def _simplify_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value

    if isinstance(value, Path):
        return value.as_posix()

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()

    if xr is not None and isinstance(value, xr.DataArray | xr.Dataset):
        try:
            return value.to_dict(data=True)
        except Exception:
            return repr(value)

    if isinstance(value, Mapping):
        return {str(k): _simplify_value(v) for k, v in value.items()}

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_simplify_value(v) for v in value]

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _simplify_value(value.to_dict())
        except Exception:
            pass

    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {str(k): _simplify_value(v) for k, v in vars(value).items()}

    return repr(value)
