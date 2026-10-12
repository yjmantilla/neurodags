"""Private 0.3.x naming rules, kept only for ``neurodags migrate``.

Nothing at runtime uses this module: since 0.4.0 derivative names are used verbatim on
disk. The migration tool needs the old rule to recognise 0.3.x output files
(``<source>@<CamelName><suffix>``) before renaming them. Not public API.
"""

from __future__ import annotations


def snake_to_camel(snake_str: str) -> str:
    """Return the 0.3.x on-disk spelling of a derivative name.

    Names without ``_`` are returned unchanged (``-`` is left as is); otherwise each
    ``_``-separated part is title-cased and joined (``fit_w_l06`` -> ``FitWL06``).

    Parameters
    ----------
    snake_str : str
        A derivative name as written in a 0.3.x ``DerivativeDefinitions``.

    Returns
    -------
    str
        The token 0.3.x wrote after ``@`` in output file names.
    """
    if "_" not in snake_str:
        return snake_str
    return "".join(part.title() for part in snake_str.split("_"))
