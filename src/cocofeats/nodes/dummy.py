from cocofeats.definitions import Artifact, NodeResult

from . import register_node


@register_node
def dummy(param1=None, param2=None) -> NodeResult:
    """
    A dummy derivative extraction function that returns a simple message.

    Parameters
    ----------
    param1 : Any, optional
        An optional parameter for demonstration purposes.
    param2 : Any, optional
        Another optional parameter for demonstration purposes.

    Returns
    -------
    NodeResult
        A NodeResult containing a simple message.
    """
    message = f"Dummy derivative extraction completed with param1={param1} and param2={param2}"

    def write_message(path: str) -> None:
        with open(path, "w") as f:
            f.write(message)

    artifacts = {".message.txt": Artifact(item=message, writer=lambda path: write_message(path))}

    return NodeResult(
        artifacts=artifacts,
    )
