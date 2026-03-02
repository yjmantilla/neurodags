# Inspired from:
# https://github.com/yjmantilla/sovabids/blob/main/tests/test_bids.py
# https://github.com/yjmantilla/sovabids/blob/main/sovabids/datasets.py


from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import mne
import numpy as np
from mne.export import export_raw

from neurodags.definitions import DatasetConfig, PathLike
from neurodags.loaders import load_configuration
from neurodags.utils import get_num_digits, get_path


def get_datasets_and_mount_point_from_pipeline_configuration(
    pipeline_input: dict[str, Any] | PathLike,
) -> dict[str, Any]:
    """
    Load the pipeline configuration and return the datasets section and mount point.
    Parameters
    ----------
    pipeline_input : dict or path-like
        Pipeline configuration as a dictionary or a path to a YAML file.
    Returns
    -------
    datasets : dict
        The datasets section of the pipeline configuration.
    mount_point : str or None
        The mount point if specified, otherwise None.
    Raises
    ------
    ValueError
        If the datasets section is missing or invalid.
    """
    pipeline_config = load_configuration(pipeline_input)
    mount_point = pipeline_config.get("mount_point", None)
    datasets = pipeline_config.get("datasets", None)
    if datasets is None:
        raise ValueError("No 'datasets' section found in the pipeline configuration.")
    if isinstance(datasets, str) or isinstance(datasets, os.PathLike):
        datasets_path = get_path(datasets, mount_point=mount_point)
        datasets = load_configuration(datasets_path)
    elif not isinstance(datasets, dict):
        raise ValueError("'datasets' section must be a dict or a path to a YAML file.")

    # At this point, datasets should be a dict
    # Make them into DatasetConfig instances
    datasets = {name: DatasetConfig(**cfg) for name, cfg in datasets.items()}

    return datasets, mount_point


def replace_brainvision_filename(fpath: PathLike, newname: str) -> None:
    """
    Replace the BrainVision file references (``DataFile`` and ``MarkerFile``) in a ``.vhdr`` header.

    This updates the entries in the ``[Common Infos]`` section so they point to
    ``<newname>.eeg`` and ``<newname>.vmrk``. Any directory components in
    ``newname`` are ignored and extensions (``.eeg`` or ``.vmrk``) are stripped.

    Parameters
    ----------
    fpath : path-like
        Path to the BrainVision header file (``.vhdr``).
    newname : str
        Base name to set for the data and marker files. If it includes
        an extension (``.eeg`` or ``.vmrk``), it will be removed.

    Returns
    -------
    None

    Notes
    -----
    - The function edits only the ``[Common Infos]`` section if present; if those
      keys aren't found there, it falls back to replacing any top-level
      ``DataFile=...`` / ``MarkerFile=...`` lines it finds.
    - Writing is done atomically via a temporary file in the same directory.
    - The function tries to decode with UTF-8 first, then falls back to Latin-1,
      which is commonly used by BrainVision headers.

    Examples
    --------
    >>> replace_brainvision_filename("recording.vhdr", "session01")
    >>> replace_brainvision_filename("recording.vhdr", "session01.eeg")  # extension stripped
    """
    path = Path(fpath)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    # Normalize newname: drop directories and strip .eeg/.vmrk (case-insensitive)
    base = os.path.basename(newname)
    base = re.sub(r"\.(eeg|vmrk)$", "", base, flags=re.IGNORECASE)

    # Read bytes; decode with UTF-8, fallback to Latin-1
    raw = path.read_bytes()
    for enc in ("utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue
    else:
        # Last-resort: replace errors (keeps file usable)
        text = raw.decode("utf-8", errors="replace")
        encoding = "utf-8"

    # Keep original line endings by splitting with keepends=True
    lines = text.splitlines(keepends=True)

    # Regex helpers
    section_re = re.compile(r"^\s*\[(?P<name>.+?)\]\s*$")
    datafile_re = re.compile(r"^\s*DataFile\s*=.*$", flags=re.IGNORECASE)
    marker_re = re.compile(r"^\s*MarkerFile\s*=.*$", flags=re.IGNORECASE)
    lineend_re = re.compile(r"(\r\n|\r|\n)$")

    def _ending(s: str) -> str:
        m = lineend_re.search(s)
        return m.group(1) if m else ""

    def _set_datafile(end: str) -> str:
        return f"DataFile={base}.eeg{end}"

    def _set_markerfile(end: str) -> str:
        return f"MarkerFile={base}.vmrk{end}"

    # First pass: replace within [Common Infos] if present
    inside_common = False
    changed = False
    for i, line in enumerate(lines):
        m = section_re.match(line)
        if m:
            inside_common = m.group("name").strip().lower() == "common infos"
            continue

        if inside_common and datafile_re.match(line):
            lines[i] = _set_datafile(_ending(line))
            changed = True
            continue
        if inside_common and marker_re.match(line):
            lines[i] = _set_markerfile(_ending(line))
            changed = True
            continue

    # Fallback: if nothing changed, replace any top-level occurrences
    if not changed:
        for i, line in enumerate(lines):
            if datafile_re.match(line):
                lines[i] = _set_datafile(_ending(line))
                changed = True
            elif marker_re.match(line):
                lines[i] = _set_markerfile(_ending(line))
                changed = True

    # Write back atomically
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, dir=str(path.parent), encoding=encoding, newline=""
        ) as tf:
            tmp_path = Path(tf.name)
            tf.writelines(lines)
        tmp_path.replace(path)
    finally:
        # Clean up if something went wrong before replace
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def make_dummy_dataset(
    EXAMPLE: PathLike | list[PathLike],
    PATTERN: str = "T%task%/S%session%/sub%subject%_%acquisition%_%run%_eeg",
    DATASET: str = "DUMMY",
    NSUBS: int = 2,
    NSESSIONS: int = 2,
    NTASKS: int = 2,
    NACQS: int = 2,
    NRUNS: int = 2,
    PREFIXES: dict[str, str] | None = None,
    ROOT: PathLike | None = None,
) -> None:
    """
    Create a dummy dataset by replicating an example file (or set of files) into a
    directory layout defined by a pattern of placeholders.

    Parameters
    ----------
    EXAMPLE : path-like or list of path-like
        Path of a file to replicate as each file in the dummy dataset. If a list,
        each item is treated as a file path and all are replicated for every
        combination generated by the pattern.
    PATTERN : str, optional
        Directory and base-filename pattern using placeholders:
        ``%dataset%``, ``%task%``, ``%session%``, ``%subject%``, ``%run%``, ``%acquisition%``.
        Forward slashes (``/``) are used as separators inside the pattern.
    DATASET : str, optional
        Name of the dataset (used to replace ``%dataset%``).
    NSUBS : int, optional
        Number of subjects.
    NSESSIONS : int, optional
        Number of sessions.
    NTASKS : int, optional
        Number of tasks.
    NACQS : int, optional
        Number of acquisitions.
    NRUNS : int, optional
        Number of runs.
    PREFIXES : dict, optional
        Mapping for prefixes with keys ``"subject"``, ``"session"``, ``"task"``,
        ``"acquisition"``, and ``"run"``. ``"run"`` is numeric in the filename, but the
        prefix is used in directory/file naming within the pattern if present.
        Defaults to ``{"subject": "SU", "session": "SE", "task": "TA", "acquisition": "AC", "run": "RU"}``.
    ROOT : path-like, optional
        Directory where files will be generated. If ``None``, uses a ``_data`` subdirectory
        relative to this module.

    Returns
    -------
    None

    Notes
    -----
    - For BrainVision files, if the example set includes ``.vhdr``/``.vmrk``, the function
      updates their internal ``DataFile``/``MarkerFile`` entries to point to the generated base name.
    - The zero-padding width for subject/session/task/acquisition/run is inferred from the
      corresponding count (e.g., ``NSUBS``) via :func:`get_num_digits`.
    - Indices start at 0 to match the original implementation.

    Examples
    --------
    Create a layout with one example file replicated across combinations:

    >>> make_dummy_dataset("example.dat", NSUBS=2, NRUNS=3)

    Replicate a BrainVision trio (.vhdr/.vmrk/.eeg) for every combination:

    >>> brains = ["template.vhdr", "template.vmrk", "template.eeg"]
    >>> make_dummy_dataset(brains, DATASET="Demo", NSUBS=1, NTASKS=1, NRUNS=2)
    """
    # Defaults
    if PREFIXES is None:
        PREFIXES = {
            "subject": "SU",
            "session": "SE",
            "task": "TA",
            "acquisition": "AC",
            "run": "RU",
        }

    # Resolve output root
    if ROOT is None:
        this_dir = Path(__file__).parent
        data_dir = (this_dir / ".." / "_data").resolve()
    else:
        data_dir = Path(ROOT)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Normalize EXAMPLE -> list[Path]
    examples: list[Path] = (
        [Path(EXAMPLE)] if not isinstance(EXAMPLE, list) else [Path(p) for p in EXAMPLE]
    )
    for ex in examples:
        if not ex.exists():
            raise FileNotFoundError(f"Example file does not exist: {ex}")

    # Build label lists (0-based indices; zero-padded lengths inferred from counts)
    sub_zeros = get_num_digits(NSUBS)
    subs = [f"{PREFIXES['subject']}{str(x).zfill(sub_zeros)}" for x in range(NSUBS)]

    task_zeros = get_num_digits(NTASKS)
    tasks = [f"{PREFIXES['task']}{str(x).zfill(task_zeros)}" for x in range(NTASKS)]

    run_zeros = get_num_digits(NRUNS)
    runs = [str(x).zfill(run_zeros) for x in range(NRUNS)]

    ses_zeros = get_num_digits(NSESSIONS)
    sessions = [f"{PREFIXES['session']}{str(x).zfill(ses_zeros)}" for x in range(NSESSIONS)]

    acq_zeros = get_num_digits(NACQS)
    acquisitions = [f"{PREFIXES['acquisition']}{str(x).zfill(acq_zeros)}" for x in range(NACQS)]

    # Generate files per combination
    for task in tasks:
        for session in sessions:
            for run in runs:
                for sub in subs:
                    for acq in acquisitions:
                        # Fill placeholders
                        dummy = (
                            PATTERN.replace("%dataset%", DATASET)
                            .replace("%task%", task)
                            .replace("%session%", session)
                            .replace("%subject%", sub)
                            .replace("%run%", run)
                            .replace("%acquisition%", acq)
                        )

                        # Resolve output path: pattern may include subdirs; last element is base name
                        parts = dummy.split("/")
                        dirpath = data_dir.joinpath(*parts[:-1])
                        dirpath.mkdir(parents=True, exist_ok=True)
                        base_out = data_dir.joinpath(*parts)  # no extension yet

                        # Copy each example, preserving extension; adjust BrainVision headers if present
                        for ex in examples:
                            ext = ex.suffix  # includes leading dot, keeps original case
                            out_fpath = Path(f"{base_out}{ext}")
                            shutil.copy2(ex, out_fpath)

                            # If copying BrainVision header/marker, update references to new base name
                            lower_ext = ext.lower()
                            if lower_ext in {".vhdr", ".vmrk"}:
                                # pass the base filename (without extension) that the header should reference
                                replace_brainvision_filename(out_fpath, parts[-1])


def generate_1_over_f_noise(
    n_channels: int,
    n_times: int,
    exponent: float = 1.0,
    *,
    sfreq: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate 1/f^alpha (pink-like) noise, suitable for synthetic EEG/MEG channels.

    The noise is produced by scaling the real FFT of white noise by ``1 / f**exponent``
    (with the DC bin set to 0), then transforming back to the time domain. Each channel
    is z-scored (zero mean, unit variance).

    Parameters
    ----------
    n_channels : int
        Number of channels to generate.
    n_times : int
        Number of time samples per channel.
    exponent : float, optional
        Spectral exponent :math:`\\alpha` in :math:`1/f^{\\alpha}`. Use 1.0 for
        “pink” noise, 0.0 for white, 2.0 for Brownian-like, etc. Default is 1.0.
    sfreq : float, optional
        Sampling frequency in Hz (used only to compute the frequency axis).
        Default is 1.0 (unit sampling).
    random_state : int | numpy.random.Generator, optional
        Seed or generator for reproducibility. If ``None``, a new Generator is used.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_channels, n_times)`` containing 1/f^alpha noise,
        z-scored independently per channel.

    Notes
    -----
    - The DC component (0 Hz) is set to 0 before inverse FFT to avoid a large
      offset when ``exponent > 0``.
    - Each channel is standardized: ``(x - mean) / std``. If a channel has zero
      variance (rare with random inputs), its standard deviation is clamped with
      a tiny epsilon to avoid division-by-zero.
    - The exact amplitude distribution is Gaussian per time point after z-scoring,
      but the spectrum follows the targeted 1/f^alpha profile in expectation.

    Examples
    --------
    >>> x = generate_1_over_f_noise(3, 10000, exponent=1.0, sfreq=250, random_state=0)
    >>> x.shape
    (3, 10000)
    >>> np.allclose(x.mean(axis=1), 0.0, atol=1e-2)
    True
    >>> np.allclose(x.std(axis=1), 1.0, atol=1e-2)
    True
    """
    if n_channels <= 0 or n_times <= 0:
        raise ValueError("n_channels and n_times must be positive integers.")

    # RNG
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    # White noise: (n_channels, n_times)
    white = rng.standard_normal((n_channels, n_times))

    # Frequency axis for rFFT (spacing d = 1/sfreq)
    # rfftfreq length is n_times//2 + 1
    freqs = np.fft.rfftfreq(n_times, d=1.0 / float(sfreq))

    # 1 / f^exponent scaling; set DC scale to 0 to avoid blow-up
    scale = np.empty_like(freqs, dtype=float)
    scale[0] = 0.0
    if exponent == 0.0:
        # No scaling (white); DC already zeroed.
        scale[1:] = 1.0
    else:
        # Avoid division by zero at DC (already set); scale others.
        scale[1:] = 1.0 / np.power(
            freqs[1:], exponent / 2.0
        )  # divide by 2 because power is squared amplitude

    # FFT along time axis, apply scale, and invert
    white_fft = np.fft.rfft(white, axis=-1)
    pink_fft = white_fft * scale[None, :]
    pink = np.fft.irfft(pink_fft, n=n_times, axis=-1)

    # Standardize per channel (mean 0, std 1) with epsilon guard
    pink -= pink.mean(axis=1, keepdims=True)
    std = pink.std(axis=1, keepdims=True)
    eps = 1e-12
    pink /= std + eps

    return pink


def get_dummy_raw(
    NCHANNELS: int = 5,
    SFREQ: float = 200.0,
    STOP: float = 10.0,
    NUMEVENTS: int = 10,
    *,
    exponent: float = 1.0,
    random_state: int | np.random.Generator | None = None,
    event_id: int = 1,
    tmin: float = -0.1,
    tmax: float = 0.4,
) -> tuple[mne.io.Raw, np.ndarray]:
    """
    Create a dummy MNE Raw object and an events array.

    The signals are 1/f^alpha (pink-like) noise per channel, z-scored independently.
    Events are placed evenly across the duration (exactly ``NUMEVENTS`` events),
    ensuring they are far enough from the start and end to allow epoching
    with the given ``tmin``/``tmax`` window.

    Parameters
    ----------
    NCHANNELS : int, optional
        Number of channels. Default is 5.
    SFREQ : float, optional
        Sampling frequency in Hz. Must be positive. Default is 200.0.
    STOP : float, optional
        Duration of the signal in seconds. Must be positive. Default is 10.0.
    NUMEVENTS : int, optional
        Number of events to generate. Must be >= 1. Default is 10.
    exponent : float, optional
        Spectral exponent ``alpha`` for the 1/f^alpha noise. Default is 1.0.
    random_state : int | numpy.random.Generator, optional
        Seed or Generator for reproducibility of the noise. Default is None.
    event_id : int, optional
        Event code to assign to all events (column 3 of the events array). Default is 1.
    tmin : float, optional
        Minimum time before event (in seconds) needed for epoching. Default is -0.1.
    tmax : float, optional
        Maximum time after event (in seconds) needed for epoching. Default is 0.4.

    Returns
    -------
    raw : mne.io.Raw
        The generated MNE Raw object with shape ``(NCHANNELS, n_times)``.
    new_events : ndarray of shape (NUMEVENTS, 3)
        The MNE-style events array: columns are (sample_index, 0, event_id).

    Notes
    -----
    - The number of samples is computed as ``n_times = int(round(SFREQ * STOP))``.
    - Events are placed at equally spaced sample indices within safe margins,
      so that each event can be used to create an epoch with the specified
      ``tmin``/``tmax`` window.

    Examples
    --------
    >>> raw, events = get_dummy_raw(NCHANNELS=3, SFREQ=250.0, STOP=5.0, NUMEVENTS=5, random_state=0)
    >>> raw.info['sfreq']
    250.0
    >>> events.shape
    (5, 3)
    """
    # ---- Validation ----
    if NCHANNELS <= 0:
        raise ValueError("NCHANNELS must be a positive integer.")
    if SFREQ <= 0:
        raise ValueError("SFREQ must be a positive number (Hz).")
    if STOP <= 0:
        raise ValueError("STOP must be a positive number (seconds).")
    if NUMEVENTS < 1:
        raise ValueError("NUMEVENTS must be >= 1.")

    # ---- Samples / time axis ----
    n_times = round(SFREQ * STOP)
    if n_times < NUMEVENTS:
        # Ensure we can place the requested number of events
        raise ValueError(
            f"Requested NUMEVENTS={NUMEVENTS} exceeds number of samples n_times={n_times}."
        )

    # ---- Channel names and info ----
    ch_names = [f"EEG{idx:03d}" for idx in range(NCHANNELS)]
    info = mne.create_info(ch_names=ch_names, sfreq=float(SFREQ), ch_types="eeg")

    # ---- Data (1/f^alpha noise, per channel) ----
    data = generate_1_over_f_noise(
        n_channels=NCHANNELS,
        n_times=n_times,
        exponent=exponent,
        sfreq=float(SFREQ),
        random_state=random_state,
    )

    # ---- Raw object ----
    raw = mne.io.RawArray(data, info)

    # ---- Events (exactly NUMEVENTS, evenly spaced, with safe margins) ----
    # Compute safe margins in samples based on tmin/tmax for later epoching
    margin_start = max(0, int(np.ceil(abs(tmin) * SFREQ)))
    margin_end = max(0, int(np.ceil(tmax * SFREQ)))

    if n_times <= (margin_start + margin_end):
        raise ValueError("Recording too short for requested margins.")

    # Place events evenly between margin_start and n_times - margin_end
    event_samples = np.linspace(
        margin_start,
        n_times - margin_end,
        NUMEVENTS,
        endpoint=False,
        dtype=int,
    )

    # Ensure uniqueness / monotonicity
    if len(np.unique(event_samples)) != NUMEVENTS:
        raise ValueError(
            f"Could not place {NUMEVENTS} unique events in {n_times} samples "
            f"with margins {margin_start}, {margin_end}."
        )

    new_events = np.column_stack(
        [event_samples, np.zeros(NUMEVENTS, dtype=int), np.full(NUMEVENTS, event_id, dtype=int)]
    )

    return raw, new_events


def get_dummy_epochs(
    NCHANNELS: int = 5,
    SFREQ: float = 200.0,
    STOP: float = 10.0,
    NUMEVENTS: int = 10,
    *,
    exponent: float = 1.0,
    random_state: int | np.random.Generator | None = None,
    event_id: int = 1,
    tmin: float = -0.1,
    tmax: float = 0.4,
    baseline: tuple[float, float] | None = (None, 0),
) -> mne.Epochs:
    """
    Create a dummy MNE Epochs object from synthetic Raw data and events.

    Parameters
    ----------
    NCHANNELS : int, optional
        Number of channels. Default is 5.
    SFREQ : float, optional
        Sampling frequency in Hz. Default is 200.0.
    STOP : float, optional
        Duration of the raw signal in seconds. Default is 10.0.
    NUMEVENTS : int, optional
        Number of events to generate. Default is 10.
    exponent : float, optional
        Spectral exponent ``alpha`` for the 1/f^alpha noise. Default is 1.0.
    random_state : int | Generator, optional
        Seed or RNG. Default is None.
    event_id : int, optional
        Event code for all events. Default is 1.
    tmin : float, optional
        Start of each epoch in seconds. Default is -0.1.
    tmax : float, optional
        End of each epoch in seconds. Default is 0.4.
    baseline : tuple or None, optional
        Baseline correction (passed to MNE). Default is (None, 0).

    Returns
    -------
    epochs : mne.Epochs
        An MNE Epochs object based on synthetic Raw data.
    """
    raw, events = get_dummy_raw(
        NCHANNELS=NCHANNELS,
        SFREQ=SFREQ,
        STOP=STOP,
        NUMEVENTS=NUMEVENTS,
        exponent=exponent,
        random_state=random_state,
        event_id=event_id,
    )

    event_dict = {"stim": event_id}
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose="error",
    )
    return epochs


def save_dummy_vhdr(fpath: PathLike, dummy_args: dict[str, Any] | None = None) -> list[Path] | None:
    """
    Save a dummy BrainVision header (``.vhdr``) plus its companion files (``.vmrk``, ``.eeg``).

    This function generates a synthetic :class:`mne.io.Raw` via :func:`get_dummy_raw`,
    then writes a BrainVision set using :func:`mne.export.export_raw`. If writing
    succeeds and all three files exist, their paths are returned; otherwise ``None`` is returned.

    Parameters
    ----------
    fpath : path-like
        Target path for the header file. If it does not end with ``.vhdr``,
        the ``.vhdr`` suffix is appended automatically.
    dummy_args : dict, optional
        Keyword arguments forwarded to :func:`get_dummy_raw` (e.g., ``NCHANNELS``,
        ``SFREQ``, ``STOP``, ``NUMEVENTS``, ``random_state``). If ``None``,
        sensible defaults from :func:`get_dummy_raw` are used.

    Returns
    -------
    list of pathlib.Path or None
        A list ``[vhdr_path, eeg_path, vmrk_path]`` if all files are successfully
        created; otherwise ``None``.

    Notes
    -----
    - The parent directory of ``fpath`` is created if it does not exist.
    - Uses the public API :func:`mne.export.export_raw` with ``fmt="brainvision"``.
      Depending on your MNE version, additional keyword arguments may be available.

    Examples
    --------
    >>> paths = save_dummy_vhdr("out/session01.vhdr", {"NCHANNELS": 4, "SFREQ": 250.0, "STOP": 2.0})
    >>> paths[0].suffix
    '.vhdr'
    """
    # Normalize and ensure .vhdr suffix
    vhdr_path = Path(fpath)
    if vhdr_path.suffix.lower() != ".vhdr":
        vhdr_path = vhdr_path.with_suffix(".vhdr")
    vhdr_path.parent.mkdir(parents=True, exist_ok=True)

    # Build dummy Raw + events
    if dummy_args is None:
        dummy_args = {}
    raw, events = get_dummy_raw(**dummy_args)

    # Attach events as annotations so export_raw will write them to .vmrk
    if events is not None and len(events):
        anns = mne.annotations_from_events(
            events=events,
            sfreq=raw.info["sfreq"],
            event_desc=None,  # default: string version of event IDs
        )
        raw = raw.set_annotations(anns)

    # Write BrainVision trio via public exporter
    # Note: export_raw determines companion .vmrk/.eeg files from the .vhdr basename
    export_raw(fname=str(vhdr_path), raw=raw, fmt="brainvision", overwrite=True)

    eeg_path = vhdr_path.with_suffix(".eeg")
    vmrk_path = vhdr_path.with_suffix(".vmrk")

    if all(p.exists() for p in (vhdr_path, eeg_path, vmrk_path)):
        return [vhdr_path, eeg_path, vmrk_path]
    return None


def generate_dummy_dataset(data_params: dict[str, Any] | None = None, generation_args: dict[str, Any] | None = None) -> None:
    """
    Generate a dummy dataset on disk.

    This function prepares an example file set (a BrainVision trio: ``.vhdr``,
    ``.vmrk``, ``.eeg``) if the caller does not supply one, then replicates it
    across a directory structure defined by a placeholder pattern using
    :func:`neurodags.datasets.make_dummy_dataset`.

    Parameters
    ----------
    data_params : dict, optional
        Parameters that control dataset generation. Recognized keys are forwarded
        to :func:`neurodags.datasets.make_dummy_dataset`, including (but not limited to):

        - ``PATTERN`` : str
            Pattern with placeholders (``%dataset%``, ``%task%``, ``%session%``,
            ``%subject%``, ``%run%``, ``%acquisition%``).
        - ``DATASET`` : str
            Dataset name (used to replace ``%dataset%``).
        - ``NSUBS``, ``NSESSIONS``, ``NTASKS``, ``NACQS``, ``NRUNS`` : int
            Grid sizes for each placeholder dimension.
        - ``PREFIXES`` : dict
            Prefix mapping (e.g., ``{"subject": "SU", "session": "SE", ...}``).
        - ``ROOT`` : path-like
            Root directory where files will be generated.
        - ``EXAMPLE`` : path-like or list of path-like
            Example file(s) to replicate. If omitted, a BrainVision trio is created
            automatically and used as the example.

    generation_args : dict, optional
        Keyword arguments forwarded to :func:`neurodags.datasets.get_dummy_raw`
        when creating the example BrainVision trio if ``EXAMPLE`` is not provided.
        Possible keys include ``NCHANNELS``, ``SFREQ``, ``STOP``, ``NUMEVENTS``,
        ``random_state``, etc. If ``None``, sensible defaults from :func:`get_dummy_raw`
        are used.
    Returns
    -------
    None
        This function performs file-system side effects only.

    Notes
    -----
    - If ``EXAMPLE`` is omitted, an example BrainVision trio is created under
      ``_data/<DATASET>/<DATASET>_template.vhdr`` and used as the source.
    - If ``ROOT`` is omitted, files are generated under
      ``_data/<DATASET>/<DATASET>_SOURCE`` relative to this module.
    - Existing ``ROOT`` is removed before new files are created.
    """
    # ---- Defaults (merged with user params) ----
    defaults: dict[str, Any] = {
        "PATTERN": "T%task%/S%session%/sub%subject%_%acquisition%_%run%",
        "DATASET": "DUMMY",
        "NSUBS": 2,
        "NTASKS": 2,
        "NRUNS": 1,
        "NSESSIONS": 1,
        "NACQS": 1,
        # PREFIXES / ROOT / EXAMPLE may be provided by caller
    }
    params = {**defaults, **(data_params or {})}

    dataset_name: str = params.get("DATASET", "DUMMY")

    # Base _data directory (…/package_root/_data)
    data_dir = (Path(__file__).parent / ".." / ".." / "_data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Default ROOT if not provided: _data/<DATASET>/<DATASET>_SOURCE
    test_root = data_dir / dataset_name
    default_root = test_root / f"{dataset_name}_SOURCE"
    root = Path(params.get("ROOT", default_root))

    # If EXAMPLE not provided, create a BrainVision trio to replicate on a tempfile
    example = params.get("EXAMPLE", None)

    with tempfile.TemporaryDirectory() as td:
        if example is None:
            tmp_vhdr = Path(td) / f"{dataset_name}_template.vhdr"
            if generation_args is None:
                generation_args = {"NCHANNELS": 2, "SFREQ": 100.0, "STOP": 10.0, "NUMEVENTS": 5}
            trio = save_dummy_vhdr(
                tmp_vhdr, dummy_args=generation_args
            )
            if not trio:
                raise RuntimeError("Failed to create example BrainVision files for dummy dataset.")
            example = trio
            params["EXAMPLE"] = example

        # Prepare output root (clean then create)
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

        # Forward to make_dummy_dataset with our resolved ROOT
        params["ROOT"] = root
        make_dummy_dataset(**params)
