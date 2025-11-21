import yaml

from cocofeats.features import register_feature_with_name
from cocofeats.loggers import get_logger

log = get_logger(__name__)

this_file = __file__
this_yaml = this_file.replace(".py", ".yml")

with open(this_yaml) as f:
    yaml_definitions = yaml.safe_load(f)


def register_features_from_dict(yaml_definitions: dict):
    for name, chain in yaml_definitions.get("FeatureDefinitions", {}).items():

        def make_wrapper(chain, name=name):  # capture both in closure
            def wrapper(
                file_path: str,
                reference_base: str | None = None,
                dataset_config=None,
                mount_point=None,
                dry_run: bool = False,
            ):
                from pathlib import Path

                from cocofeats.dag import run_feature

                return run_feature(
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
        # pass chain as definition so FeatureEntry stores it
        register_feature_with_name(name, func, definition=chain, override=True)
        log.info("Registered feature", name=name)


register_features_from_dict(yaml_definitions)
