# conf.py
import importlib
import inspect
import os
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v
from pathlib import Path

project = "neurodags"

# Full version from installed dist (hatch-vcs stamps this)
try:
    release = _v("neurodags")
except PackageNotFoundError:
    release = "0+unknown"

# Short X.Y
version = ".".join(release.split(".")[:2])


def _git_short() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return ""


_commit = _git_short()
html_title = f"{project} {release}" + (f" ({_commit})" if _commit else "")

# If you use a src/ layout, add it to sys.path for things like sphinx.ext.autodoc (AutoAPI doesn't need it)
sys.path.insert(0, os.path.abspath("../src"))

author = "Yorguin-José Mantilla-Ramos"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

# -- General configuration ------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    #    "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
    "autoapi.extension",
]

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # AutoAPI will still crawl ../src; these patterns are for the docs/ tree
]

# -- AutoAPI configuration -----------------------------------------------
# Point to your actual source root (adjust if your package lives elsewhere)
autoapi_type = "python"
autoapi_dirs = ["../src"]  # or ["../src/neurodags"] to target just the package
autoapi_root = "api"  # generated top-level folder under docs
autoapi_add_toctree_entry = True  # add an "API Reference" entry automatically
autoapi_generate_api_docs = True
autoapi_member_order = "bysource"  # preserve source order
autoapi_python_use_implicit_namespaces = True  # if you have any namespace pkgs
autoapi_python_class_content = "both"  # class + __init__ docs
autoapi_keep_files = True  # useful for debugging

# Be generous so you truly get “the whole API”
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",  # include _private
    "special-members",  # e.g. __call__, __iter__
    # "inherited-members",
    # "show-inheritance",
    "show-module-summary",
    # "imported-members",   # enable if you also want re-exported/imported names
]

def skip_troublesome_tui_members(app, what, name, obj, skip, options):
    # Skip CSS and BINDINGS which cause formatting issues
    if any(x in name for x in ("DEFAULT_CSS", "CSS", "BINDINGS")):
        return True
    # Skip private tab classes entirely as they inherit problematic docstrings from textual
    if any(
        x in name
        for x in (
            "_ConfigTab",
            "_DagTab",
            "_DryRunTab",
            "_RunTab",
            "_DataFrameTab",
            "_NcTab",
        )
    ):
        return True
    # Skip NeuroDagsApp.__init__ to avoid inherited textual docstrings
    if what == "method" and name.endswith(".NeuroDagsApp.__init__"):
        return True
    return skip

def setup(app):
    app.connect("autoapi-skip-member", skip_troublesome_tui_members)

# Ignore docs generation for tests/examples if desired (pattern applies to source crawl)
autoapi_ignore = [
    "*/tests/*",
    "*_test.py",
    "*/conftest.py",
]

# -- Options for HTML output ----------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/yjmantilla/neurodags",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/yjmantilla/neurodags",
            "html": """<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>""",
            "class": "",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/neurodags/",
            "html": """<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 640 512" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M425.7 256c-16.9 0-32.8-9-41.4-23.4L320 126l-64.2 106.6c-8.7 14.5-24.6 23.5-41.5 23.5-4.5 0-9-.6-13.3-1.9L64 215v178c0 14.7 10 27.5 24.2 31l216.2 54.1c10.2 2.5 20.9 2.5 31 0L551.8 424c14.2-3.6 24.2-16.4 24.2-31V215l-136.9 39.1c-4.3 1.3-8.8 1.9-13.4 1.9zm212.6-112.2L586.8 41c-3.1-6.2-9.8-9.8-16.7-8.9L320 64l91.7 152.1c3.8 6.3 11.4 9.3 18.5 7.3l197.9-56.5c9.9-2.9 14.4-14.2 10.2-23.1zM53.2 41L1.7 143.8c-4.3 8.9.3 20.2 10.1 23l197.9 56.5c7.1 2 14.7-1 18.5-7.3L320 64 69.8 32.1c-6.9-.9-13.5 2.7-16.6 8.9z"/></svg>""",
            "class": "",
        },
    ],
}

# MyST settings
myst_enable_extensions = [
    "deflist",
    "fieldlist",
    "colon_fence",
]

# Sphinx-gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": ["examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r"/plot_.*\.py$",
    "download_all_examples": False,
}


# Adjust these to your org/repo/default branch
GITHUB_USER = "yjmantilla"
GITHUB_REPO = "neurodags"
GITHUB_BRANCH = "main"  # or "master"


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    modname = info.get("module")
    fullname = info.get("fullname")
    if not modname:
        return None
    try:
        mod = importlib.import_module(modname)
    except Exception:
        return None
    obj = mod
    for part in (fullname or "").split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            break

    try:
        fn = inspect.getsourcefile(obj) or inspect.getfile(obj)
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        try:
            fn = inspect.getsourcefile(mod) or inspect.getfile(mod)
            source, lineno = inspect.getsourcelines(mod)
        except Exception:
            return None

    # Make path relative to repo root
    fn = os.path.relpath(fn, start=Path(__file__).resolve().parents[1])

    linespec = f"#L{lineno}-L{lineno+len(source)-1}"
    return f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{fn}{linespec}"


# # --- Hide Pydantic internals from docs ---------------------------------------

# _PYDANTIC_INTERNAL_ATTRS = {
#     "model_config",
#     "__class_vars__",
#     "__private_attributes__",
#     "__signature__",
#     "__pydantic_complete__",
#     "__pydantic_core_schema__",
#     "__pydantic_custom_init__",
#     "__pydantic_decorators__",
#     "__pydantic_generic_metadata__",
#     "__pydantic_parent_namespace__",
#     "__pydantic_post_init__",
#     "__pydantic_root_model__",
#     "__pydantic_serializer__",
#     "__pydantic_validator__",
#     "__pydantic_fields__",
#     "__pydantic_setattr_handlers__",
#     "__pydantic_computed_fields__",
# }


# def _is_pydantic_internal(name: str, obj: object) -> bool:
#     # Any of the explicit ones
#     if name in _PYDANTIC_INTERNAL_ATTRS:
#         return True

#     # Anything starting with this prefix
#     if name.startswith("__pydantic_"):
#         return True

#     # Extra safety: things whose defining module is clearly pydantic
#     mod = getattr(obj, "__module__", "") or ""
#     if mod.startswith("pydantic") or mod.startswith("pydantic_core"):
#         return True

#     return False


# def autodoc_skip_pydantic(app, what, name, obj, skip, options):
#     if _is_pydantic_internal(name, obj):
#         return True  # skip it
#     return None      # use default behaviour otherwise


# def autoapi_skip_pydantic(app, what, name, obj, skip, options):
#     if _is_pydantic_internal(name, obj):
#         return True  # skip it in AutoAPI too
#     return None

# def setup(app):
#     app.connect("autodoc-skip-member", autodoc_skip_pydantic)
#     app.connect("autoapi-skip-member", autoapi_skip_pydantic)
