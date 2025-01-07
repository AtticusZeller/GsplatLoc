import json
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd

data_set = Literal["Replica", "TUM"]
room = Literal[
    "room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"
]
exp_methods = Literal["ICP", "GICP", "PLANE_ICP", "HYBRID", "ours"]
error = Literal["ATE", "AAE"]


def convert_method_name(method: exp_methods) -> str:
    """
    Convert the internal method name to its display name.

    :param method: Internal method name (e.g., "ICP", "GICP").
    :return: Display name for the method.
    """
    method_mapping = {
        "ICP": "RTG-SLAM(ICP)",
        "GICP": "GS-ICP-SLAM(GICP)",
        "PLANE_ICP": "Gaussian-SLAM(PLANE ICP)",
        "HYBRID": "Gaussian-SLAM(HYBRID)",
        "ours": "Ours",
    }
    return method_mapping[method]


def process_replica_data(
    data: dict[data_set, dict[room, dict[exp_methods, dict[error, float]]]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process Replica dataset to calculate ATE and AAE results.

    :param data: Nested dictionary containing Replica dataset results.
    :return: ATE and AAE results as pandas DataFrames.
    """
    # Define sequence order
    sequence_order: list[room] = [
        "room0",
        "room1",
        "room2",
        "office0",
        "office1",
        "office2",
        "office3",
        "office4",
    ]
    column_names = ["Avg.", "R0", "R1", "R2", "Of0", "Of1", "Of2", "Of3", "Of4"]

    # Initialize result dictionaries
    ate_results: dict[str, list[float]] = {}
    aae_results: dict[str, list[float]] = {}

    methods: list[exp_methods] = ["ICP", "GICP", "PLANE_ICP", "HYBRID", "ours"]
    for method in methods:
        # value sequence under the method
        ate_values: list[float] = []
        aae_values: list[float] = []

        # Collect data for each sequence
        for seq in sequence_order:
            if seq in data["Replica"]:
                seq_data = data["Replica"][seq]
                if method in seq_data:
                    # Convert ATE to centimeters
                    ate_values.append(seq_data[method]["ATE"] * 100)
                    aae_values.append(seq_data[method]["AAE"])

        # Compute averages and add to results
        method_name = convert_method_name(method)
        ate_avg = float(np.mean(ate_values))
        aae_avg = float(np.mean(aae_values))

        # Concatenation ate_avg to head
        ate_results[method_name] = [ate_avg] + ate_values
        aae_results[method_name] = [aae_avg] + aae_values

    # Create DataFrames
    ate_df = pd.DataFrame(ate_results, index=column_names).T
    aae_df = pd.DataFrame(aae_results, index=column_names).T

    return ate_df, aae_df


def process_tum_data(
    data: dict[str, dict[str, dict[str, dict[str, float]]]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process TUM dataset to calculate ATE and AAE results.

    :param data: Nested dictionary containing TUM dataset results.
    :return: ATE and AAE results as pandas DataFrames.
    """
    # Define sequence mapping
    sequence_mapping = {
        "freiburg1_desk": "fr1/desk",
        "freiburg1_desk2": "fr1/desk2",
        "freiburg1_room": "fr1/room",
        "freiburg2_xyz": "fr2/xyz",
        "freiburg3_long_office_household": "fr3/off.",
    }

    column_names = ["Avg."] + list(sequence_mapping.values())

    # Initialize result dictionaries
    ate_results: dict[str, list[float]] = {}
    aae_results: dict[str, list[float]] = {}

    methods: list[exp_methods] = ["ICP", "GICP", "PLANE_ICP", "HYBRID", "ours"]

    for method in methods:
        ate_values = []
        aae_values = []

        # Collect data for each sequence
        for seq in sequence_mapping.keys():
            if seq in data["TUM"]:
                seq_data = data["TUM"][seq]
                if method in seq_data:
                    # Convert ATE to centimeters
                    ate_values.append(seq_data[method]["ATE"] * 100)
                    aae_values.append(seq_data[method]["AAE"])

        # Compute averages and add to results
        method_name = convert_method_name(method)
        ate_avg = float(np.mean(ate_values))
        aae_avg = float(np.mean(aae_values))

        ate_results[method_name] = [ate_avg] + ate_values
        aae_results[method_name] = [aae_avg] + aae_values

    # Create DataFrames
    ate_df = pd.DataFrame(ate_results, index=column_names).T
    aae_df = pd.DataFrame(aae_results, index=column_names).T

    return ate_df, aae_df


def format_table(df: pd.DataFrame, metric_name: str) -> None:
    """
    Format and print a table with ATE or AAE results.

    :param df: DataFrame containing results.
    :param metric_name: Name of the metric (e.g., "ATE RMSE ↓[cm]").
    """
    print(f"\n{metric_name}:")
    print("-" * 50)
    for idx, row in df.iterrows():
        values = [f"{x:.5f}" for x in row]
        print(f"{idx:<60} {'|'.join(values)}")
    print("-" * 50)


def main() -> None:
    """
    Main function to process the datasets and print results.
    """
    # Load JSON data
    with open("res.json") as f:
        data = json.load(f)

    # Process Replica dataset
    replica_ate_df, replica_aae_df = process_replica_data(data)
    print("\nReplica Dataset Results:")
    format_table(replica_ate_df, "ATE RMSE ↓[cm]")
    format_table(replica_aae_df, "AAE RMSE ↓[°]")

    # Process TUM dataset
    tum_ate_df, tum_aae_df = process_tum_data(data)
    print("\nTUM Dataset Results:")
    format_table(tum_ate_df, "ATE RMSE ↓[cm]")
    format_table(tum_aae_df, "AAE RMSE ↓[°]")


if __name__ == "__main__":
    main()
