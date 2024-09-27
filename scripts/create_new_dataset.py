import argparse
from typing import List

from melo_benchmark.data_processing.dataset_builder import (
    DatasetBuilder,
    SUPPORTED_CROSSWALKS
)
from melo_benchmark.data_processing.esco_loader import SUPPORTED_ESCO_LANGUAGES


valid_sources = SUPPORTED_CROSSWALKS.keys()
valid_languages = SUPPORTED_ESCO_LANGUAGES


def validate_source_taxonomy(value: str) -> str:
    if value not in valid_sources:
        x = ', '.join(valid_sources)
        raise argparse.ArgumentTypeError(
            f"Invalid source taxonomy '{value}'. Valid options are: {x}."
        )
    return value


def validate_target_languages(value: str) -> List[str]:
    languages = value.split(",")  # Split the input into a list
    for lang in languages:
        if lang not in valid_languages:
            x = ', '.join(valid_languages)
            raise argparse.ArgumentTypeError(
                f"Invalid target language '{lang}'. Valid options are: {x}."
            )
    return languages


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process taxonomy and target languages."
    )

    help_m = f"Source taxonomy. Must be one of: {', '.join(valid_sources)}."
    parser.add_argument(
        '--source_taxonomy',
        type=validate_source_taxonomy,
        required=True,
        help=help_m
    )

    help_m = "Target languages as a comma-separated list. " \
             + "e.g. `--target_languages en,es,de`. " \
             + f"Valid options are: {', '.join(valid_languages)}"
    parser.add_argument(
        '--target_languages',
        type=validate_target_languages,
        required=True,
        help=help_m
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    source_taxonomy = args.source_taxonomy
    target_languages = args.target_languages

    dataset_builder = DatasetBuilder(source_taxonomy)

    output_path = dataset_builder.build(target_languages)

    print(f"\nThe new dataset was created:\n{output_path}\n")
