#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact
"""
import argparse
import logging
from tempfile import TemporaryDirectory
from pandas.core.frame import DataFrame
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Download from W&B the raw dataset and apply some basic data cleaning,
    exporting the result to a new artifact
    """
    logger.info("Starting cleaning")
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Downloading artifact: %s", args.input_artifact)

    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)
    if not isinstance(df, DataFrame):
        logger.error("Failed to read artifact: %s", artifact_local_path)
        raise ValueError(f"Expected a DataFrame, got {type(df)}")

    logger.info("Cleaning data")

    range_prices = df['price'].between(args.min_price, args.max_price)
    range_lng = df['longitude'].between(-74.25, -73.50)
    range_lat = df['latitude'].between(40.5, 41.2)
    idx = range_prices & range_lng & range_lat
    df = df[idx].copy()

    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info("Exporting artifact: %s", args.output_artifact)

    with TemporaryDirectory() as tmpdir:
        filename = "cleaned.csv"
        tmp_file = tmpdir + "/" + filename
        df.to_csv(tmp_file, index=False)
        artifact = wandb.Artifact(
            args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )

        artifact.add_file(tmp_file)
        run.log_artifact(artifact)

        artifact.wait()

    logger.info("Done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="The artifact to download",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="The artifact to upload",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="The type of artifact to upload",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="The description of the artifact to upload",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="The minimum price to keep",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="The maximum price to keep",
        required=True
    )

    args = parser.parse_args()

    go(args)
