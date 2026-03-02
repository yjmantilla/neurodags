import yaml

from cocofeats.derivatives import register_derivative_with_name
from cocofeats.loggers import get_logger

log = get_logger(__name__)

this_file = __file__
this_yaml = this_file.replace(".py", ".yml")

with open(this_yaml) as f:
    yaml_definitions = yaml.safe_load(f)


def register_derivatives_from_dict(yaml_definitions: dict):
    for name, chain in yaml_definitions.get("DerivativeDefinitions", {}).items():

        def make_wrapper(chain, name=name):  # capture both in closure
            def wrapper(
                file_path: str,
                reference_base: str | None = None,
                dataset_config=None,
                mount_point=None,
                dry_run: bool = False,
            ):
                from pathlib import Path

                from cocofeats.dag import run_derivative

                return run_derivative(
                    chain,
                    name,
                    file_path,
                    reference_base=Path(reference_base or ""),
                    dataset_config=dataset_config,
                    mount_point=mount_point,
                    dry_run=dry_run,
                )

            return wrapper

        func = make_wrapper(chain)
        # pass chain as definition so DerivativeEntry stores it
        register_derivative_with_name(name, func, definition=chain, override=True)
        log.info("Registered derivative", name=name)


register_derivatives_from_dict(yaml_definitions)
