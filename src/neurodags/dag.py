# dag_exec.py
import glob
import inspect
import json
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

import yaml

from neurodags.definitions import NodeResult
from neurodags.derivatives import (
    get_derivative as get_derivative_definition,
)
from neurodags.derivatives import (
    list_derivatives as list_derivative_definitions,
)
from neurodags.derivatives import (
    register_derivative_with_name,
)
from neurodags.loggers import get_logger
from neurodags.nodes import get_node
from neurodags.utils import snake_to_camel

log = get_logger(__name__)
_ID_REF = re.compile(r"^id\.(\d+)$")

# ---------- helpers ----------


def _is_id_ref(x: Any) -> bool:
    return isinstance(x, str) and _ID_REF.match(x) is not None


def _resolve_refs(obj: Any, store: dict[int, Any]) -> Any:
    if _is_id_ref(obj):
        sid = int(_ID_REF.match(obj).group(1))
        if sid not in store:
            raise KeyError(f"Reference {obj} not computed yet")
        return store[sid]
    if isinstance(obj, dict):
        return {k: _resolve_refs(v, store) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        t = type(obj)
        return t(_resolve_refs(v, store) for v in obj)
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


def _prep_kwargs(raw_kwargs: dict[str, Any], store: dict[int, Any]) -> dict[str, Any]:
    resolved = _resolve_refs(raw_kwargs or {}, store)
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


# Optional: simple cache probe (customize for your FS layout)
def _is_step_cached(
    derivative_name: str, step_derivative_name: str, reference_base: Path, accept_ambiguous=False
) -> bool:
    """
    Decide if a 'derivative' step can be considered already computed.
    By default: look for any file whose prefix matches:
    <reference_base>@<CamelCase(step_derivative_name)>.*
    """
    # prefix = f"{reference_base}@{snake_to_camel(step_derivative_name)}"
    postfix = f"@{step_derivative_name}"
    # return any(reference_base.parent.glob(Path(prefix).name + ".*"))
    candidates = glob.glob(
        reference_base.as_posix() + postfix
        if isinstance(reference_base, Path)
        else reference_base + postfix
    )  # or use
    if len(candidates) > 1:
        if not accept_ambiguous:
            log.error(
                "Multiple cached artifacts found",
                derivative=derivative_name,
                step_reference=step_derivative_name,
                reference_base=reference_base,
                candidates=candidates,
            )
            raise RuntimeError(
                f"Multiple cached artifacts found for {reference_base}{postfix}: {candidates}"
            )
        else:
            log.warning(
                "Multiple cached artifacts found, using the first one",
                derivative=derivative_name,
                step_reference=step_derivative_name,
                reference_base=reference_base,
                candidates=candidates,
            )
    return len(candidates) >= 1, candidates[0] if candidates else None


# --- registration from YAML ---
def register_derivatives_from_yaml(yaml_path: str) -> list[str]:
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
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
        log.info("Registered derivative", name=derivative_name)

    return registered


def _artifact_candidates_for(prefix: str) -> list[str]:
    # TODO: How to add option to skip derivatives that previously failed? (.error files)
    return [x for x in sorted(glob.glob(prefix + ".*")) if not x.endswith(".error")]


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
) -> NodeResult | dict[str, Any]:
    """
    Evaluate (or dry-run) a derivative on a single file.

    When dry_run=True:
      - Do NOT execute any steps.
      - Report, per step, whether the expected artifacts are already present.
      - Return a dict with a 'plan' list (so callers don't confuse it with NodeResult).

    Normal mode:
      - 'derivative' steps: if cached and overwrite=False, skip compute; else compute or recurse.
      - 'node' steps: execute; if it's the last step, save artifacts under '@<DerivativeName>*'.
    """
    overwrite = bool(derivative_def.get("overwrite", False))
    save = bool(derivative_def.get("save", True))
    nodes = derivative_def.get("nodes", [])
    order = _topo_order(nodes)

    store: dict[int, Any] = {}
    last_result: NodeResult | None = None
    plan: list[dict[str, Any]] = [] if dry_run else None

    def _record(**kwargs):
        if plan is not None:
            plan.append(kwargs)

    # In dry-run, also check the final output of the derivative upfront
    base_reference = (
        reference_base.as_posix() if isinstance(reference_base, Path) else reference_base
    )
    final_prefix = base_reference + "@" + snake_to_camel(derivative_name) if save else None
    if dry_run:
        if save:
            final_candidates = _artifact_candidates_for(final_prefix)
            _record(
                id="final",
                kind="derivative_output",
                name=derivative_name,
                prefix=final_prefix,
                cached=len(final_candidates) > 0,
                paths=final_candidates,
            )
        else:
            _record(
                id="final",
                kind="derivative_output",
                name=derivative_name,
                will_save=False,
                cached=False,
                paths=[],
            )
    else:
        # Early skip if final already cached and not overwriting
        if save and not overwrite:
            final_candidates = _artifact_candidates_for(final_prefix)
            if len(final_candidates) > 0:
                log.info(
                    "Last step of the derivative is cached",
                    derivative=derivative_name,
                    final_prefix=final_prefix,
                    reference_base=reference_base,
                    candidate=final_candidates[0],
                )
                return {"cached": final_candidates}

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
            prefix = (
                reference_base.as_posix() + "@" + snake_to_camel(base_name)
                if isinstance(reference_base, Path)
                else reference_base + "@" + snake_to_camel(base_name)
            )
            candidates = _artifact_candidates_for(prefix)
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
                    )
                _record(**entry)
                continue

            # normal execution path
            cached_here = (len(candidates) > 0) and not overwrite
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
                sub_result = run_derivative(
                    get_derivative_definition(base_name).definition,
                    base_name,
                    file_path,
                    reference_base=reference_base,
                    dataset_config=dataset_config,
                    mount_point=mount_point,
                    dry_run=False,
                )
                # sub-derivative may return None if already cached
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

            kwargs = _prep_kwargs(step.get("args", {}), store)
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
                    # save final artifacts under '@<DerivativeName>'
                    assert final_prefix is not None
                    for artifact_name, artifact in res.artifacts.items():
                        try:
                            log.info("Processed artifact", name=artifact_name, file=artifact)
                            artifact_path = final_prefix + artifact_name
                            artifact.writer(artifact_path)
                            log.info("Saved artifact", file=artifact_path)
                        except Exception as e:
                            log.error(
                                "Error saving artifact",
                                artifact_name=artifact_name,
                                path=final_prefix + artifact_name,
                                error=str(e),
                                exc_info=True,
                            )
                            raise
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
                        error_path = final_prefix + ".error"
                        with open(error_path, "w", encoding="utf-8") as ef:
                            ef.write(
                                f"Derivative '{derivative_name}' step id={sid} node='{node_name}' failed:\n{e!s}\n"
                            )
                        log.info("Wrote error marker file", path=error_path)
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
            "reference_base": (
                reference_base.as_posix() if isinstance(reference_base, Path) else reference_base
            ),
            "overwrite": overwrite,
            "plan": plan,
        }

    if last_result is None:
        raise RuntimeError(f"Derivative '{derivative_name}' produced no result")
    return last_result


def collect_derivative_for_dataframe(
    derivative_def: dict,
    derivative_name: str,
    file_path: str,
    reference_base: Path,
    dataset_config: Any = None,
    mount_point: Path | None = None,
    *,
    flatten_xarray_artifacts: bool = True,
    sort_flattened_dims: bool = True,
    preserve_complex_values: bool = False,
) -> dict[str, Any]:
    """
    Collect artifacts for a derivative and convert them into dataframe-ready values.

    This helper prefers cached artifacts if they exist on disk. If none are available,
    it will execute ``run_derivative`` to obtain the necessary results (without forcing a recompute
    when cached outputs are already present).

    Parameters
    ----------
    flatten_xarray_artifacts:
        When True, ``xr.DataArray`` artifacts (or paths to ``.nc`` files containing them)
        are flattened into multiple columns using their coordinate metadata.
    sort_flattened_dims:
        When flattening a ``DataArray``, sort dimension names alphanumerically when constructing
        the column suffix.

    preserve_complex_values:
        When True, skip flattening and value simplification so artifacts retain their original
        Python objects (after potential on-disk loading).

    Returns a dictionary suitable for composing a DataFrame row, with columns named
    ``{derivative_name}{artifact_suffix}`` (with optional BIDS-like suffixes per flattened element).
    Derivatives flagged with ``save=True`` and ``for_dataframe=True`` are skipped when no cached
    artifacts are found instead of triggering a recomputation.
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
        )
    except _MissingPrecomputedArtifacts:
        log.debug(
            "Skipping derivative during dataframe collection because cached artifacts are missing",
            derivative=derivative_name,
            file=file_path,
            reference_base=(
                reference_base.as_posix() if isinstance(reference_base, Path) else reference_base
            ),
        )
        return {}

    row_values: dict[str, Any] = {}
    for suffix, payload in artifact_payloads.items():
        column_name = f"{derivative_name}{suffix}"
        simplified = _simplify_artifact_payload(
            payload,
            suffix=suffix,
            flatten_xarray=flatten_xarray_artifacts and not preserve_complex_values,
            sort_dims_alphabetically=sort_flattened_dims,
            preserve_complex_values=preserve_complex_values,
        )
        if isinstance(simplified, _FlattenedArtifactColumns):
            for flatten_suffix, value in simplified.items():
                target_column = (
                    column_name if flatten_suffix == "" else f"{column_name}@{flatten_suffix}"
                )
                if target_column in row_values:
                    log.warning(
                        "Overwriting dataframe column while flattening artifact",
                        column=target_column,
                        derivative=derivative_name,
                        suffix=suffix,
                    )
                row_values[target_column] = value
        else:
            row_values[column_name] = simplified
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
) -> dict[str, Any]:
    save = bool(derivative_def.get("save", True))
    base_reference = (
        reference_base.as_posix() if isinstance(reference_base, Path) else reference_base
    )
    prefix = base_reference + "@" + snake_to_camel(derivative_name)

    payloads: dict[str, Any] = {}
    if save:
        cached_paths = _artifact_candidates_for(prefix)
        if cached_paths:
            for path in cached_paths:
                suffix = path[len(prefix) :]
                payloads[suffix] = Path(path)
            return payloads
        if precomputed_only:
            raise _MissingPrecomputedArtifacts(
                f"Cached artifacts not found for derivative '{derivative_name}' with prefix '{prefix}'"
            )

    result = run_derivative(
        derivative_def,
        derivative_name,
        file_path,
        reference_base,
        dataset_config=dataset_config,
        mount_point=mount_point,
        dry_run=False,
    )

    if isinstance(result, NodeResult):
        for suffix, artifact in result.artifacts.items():
            payload: Any = artifact.item
            if save:
                artifact_path = prefix + suffix
                if Path(artifact_path).exists():
                    payload = Path(artifact_path)
            payloads[suffix] = payload
        return payloads

    if isinstance(result, dict) and "cached" in result:
        paths = result["cached"]
        for path in paths:
            suffix = path[len(prefix) :]
            payloads[suffix] = Path(path)
        return payloads

    raise RuntimeError(
        f"Unexpected result type when collecting derivative '{derivative_name}': {type(result)!r}"
    )


def _simplify_artifact_payload(
    payload: Any,
    *,
    suffix: str,
    flatten_xarray: bool = False,
    sort_dims_alphabetically: bool = True,
    preserve_complex_values: bool = False,
) -> Any:
    value = _load_from_path(payload, suffix=suffix) if isinstance(payload, Path) else payload

    if preserve_complex_values:
        return value

    if flatten_xarray and xr is not None and isinstance(value, xr.DataArray):
        return _flatten_dataarray_payload(value, sort_dims_alphabetically=sort_dims_alphabetically)

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
    """Marker mapping used to expand flattened DataArray payloads into multiple columns."""


def _flatten_dataarray_payload(
    data_array: "xr.DataArray", *, sort_dims_alphabetically: bool
) -> _FlattenedArtifactColumns:
    if data_array.ndim == 0:
        try:
            scalar = data_array.item()
        except Exception:
            scalar = data_array.values
        return _FlattenedArtifactColumns({"": _simplify_value(scalar)})

    dims = list(data_array.dims)
    shape = tuple(data_array.sizes[dim] for dim in dims)
    index_iterator = _ndindex(shape)

    flattened: dict[str, Any] = {}
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
            dims_with_values.append((dim, coord_value))

        if sort_dims_alphabetically:
            dims_with_values.sort(key=lambda item: item[0])

        # Build a BIDS-like segment: dim-dimvalue pairs joined by underscores.
        key = "_".join(
            f"{dim}-{_format_coord_value(coord_value)}" for dim, coord_value in dims_with_values
        )

        try:
            raw_value = data_array.data[index_tuple]
        except Exception:
            raw_value = data_array.values[index_tuple]

        flattened[key] = _simplify_value(raw_value)

    return _FlattenedArtifactColumns(flattened)


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


def _format_coord_value(value: Any) -> str:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    "NA" if value is None else str(value)
    # sanitized = re.sub(r"[^0-9A-Za-z\-.@~$]+", "-", text)
    # sanitized = sanitized.strip("-")
    value = str(value)

    if isinstance(value, str):
        value = value.replace("-", "~")
        value = value.replace("_", "|")
        value = value.replace("@", "$")
        value = value.replace(",", ".")
    sanitized = value

    return sanitized or "NA"


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
