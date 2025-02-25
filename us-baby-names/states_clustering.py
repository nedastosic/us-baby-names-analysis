import pandas as pd
import numpy as np
import csv
import ast
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from kmodes.kmodes import KModes


def load_and_merge_data(state_file: str, national_file: str) -> pd.DataFrame:
    """
    Loads state and national data, merges on Name, Gender, and Year,
    renames count columns, and computes the state-to-national ratio.
    """
    df_states = pd.read_csv(state_file)
    df_national = pd.read_csv(national_file)

    # Merge on Name, Gender, and Year
    df_merged = pd.merge(
        df_states, df_national,
        on=["Name", "Gender", "Year"],
        suffixes=('_state', '_national')
    )

    # Rename count columns for clarity
    df_merged.rename(
        columns={'Count_state': 'state_count', 'Count_national': 'national_count'},
        inplace=True
    )

    # Replace zeros in national_count to avoid division by zero and compute ratio
    df_merged['national_count'] = df_merged['national_count'].replace(0, np.nan)
    df_merged['ratio'] = df_merged['state_count'] / df_merged['national_count']

    return df_merged


def pivot_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivots the merged DataFrame so that each row represents a name and
    each column represents a state (with the average ratio over time).
    """
    # Aggregate by Name and State (average ratio over all years)
    df_grouped = df.groupby(["Name", "State"])["ratio"].mean().reset_index()
    df_grouped.to_csv("./data/JoinedNamesGroupedByState.csv", index=False)
    # Pivot so that rows are names and columns are states; fill missing with 0.
    df_pivot = df_grouped.pivot(index="Name", columns="State", values="ratio").fillna(0)
    return df_pivot


def find_significant_names(df_pivot: pd.DataFrame, factor: float = 2.0, min_diff: float = 0.5) -> pd.DataFrame:
    """
    For each name (row in the pivoted DataFrame), identifies the states where the
    ratio is significantly higher than the name's overall average.

    A state is flagged if its ratio is more than `factor` times the mean ratio for that name.
    Only names that have a difference between the maximum and minimum ratio greater than
    min_diff are reported.

    Returns a DataFrame with columns: name and significant_states (list of states).
    """
    significant = []
    # Iterate over each name (row) in the pivot table
    for name, row in df_pivot.iterrows():
        mean_ratio = row.mean()
        # Only consider names with a sufficiently large range of ratios
        if row.max() - row.min() > min_diff:
            # Find states where the ratio is greater than factor * mean_ratio
            sig_states = row[row > factor * mean_ratio].index.tolist()
            if sig_states:
                significant.append({"name": name, "significant_states": sig_states})
    return pd.DataFrame(significant)


def cluster_states(csv_file: str, n_clusters: int = 2):
    """
    Loads the significant names CSV, builds a binary matrix for each name and its
    significant states, transposes it so that rows are states and columns are names,
    clusters the states using K-Modes clustering, and plots the clusters using PCA.
    """
    # Read the CSV file (using semicolon as delimiter)
    df = pd.read_csv(csv_file, sep=";")

    # Parse the 'significant_states' column into a list
    def parse_states(s):
        try:
            # Convert string representation of list to an actual list
            return ast.literal_eval(s)
        except Exception:
            s = s.strip('[]')
            return [state.strip() for state in s.split(',') if state.strip()]

    df['parsed_states'] = df['significant_states'].apply(parse_states)

    # Create a sorted list of all unique states from the parsed lists
    all_states = sorted({state for states in df['parsed_states'] for state in states})

    # Build a binary matrix where rows represent names and columns represent states
    matrix = pd.DataFrame({
        state: df['parsed_states'].apply(lambda states: 1 if state in states else 0)
        for state in all_states
    })
    matrix.index = df['name']

    # Transpose the matrix so that rows are states (features are names)
    state_matrix = matrix.T

    # Apply K-Modes clustering
    km = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=1)
    clusters = km.fit_predict(state_matrix)
    state_matrix['cluster'] = clusters

    # Group and print states by their cluster assignment
    clusters_dict = {}
    for state, cluster in zip(state_matrix.index, clusters):
        clusters_dict.setdefault(cluster, []).append(state)

    print("State clusters based on similar appeal across names:")
    for cluster, states in clusters_dict.items():
        print(f"Cluster {cluster}: {states}")

    # --- Plotting using PCA ---
    # Drop the cluster column to extract features for PCA
    features = state_matrix.drop('cluster', axis=1)
    pca = PCA(n_components=2)
    components = pca.fit_transform(features)
    state_matrix['pca1'] = components[:, 0]
    state_matrix['pca2'] = components[:, 1]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(state_matrix['pca1'], state_matrix['pca2'],
                          c=state_matrix['cluster'], cmap='viridis', s=100)
    plt.title("State Clusters (PCA Projection)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, label='Cluster')
    for i, state in enumerate(state_matrix.index):
        plt.annotate(state, (state_matrix['pca1'].iloc[i], state_matrix['pca2'].iloc[i]),
                     fontsize=9, ha='right')
    plt.tight_layout()
    plt.show()

    return state_matrix


def main():
    state_file = "data/StateNames.csv"
    national_file = "data/NationalNames.csv"

    # Step 1: Load and merge the data
    df_merged = load_and_merge_data(state_file, national_file)
    merged_output = "./data/JoinedNamesOverTime.csv"
    df_merged.to_csv(merged_output, index=False)
    print(f"Merged data saved to {merged_output}")

    # Step 2: Pivot the data (names as rows, states as columns with average ratio)
    df_pivot = pivot_data(df_merged)

    # Step 3: Identify names that are significantly higher in some states
    significant_names = find_significant_names(df_pivot, factor=2.0, min_diff=0.5)

    # Convert list of significant states to a string formatted as [NY, CA] with no extra quotes
    significant_names['significant_states'] = significant_names['significant_states'].apply(
        lambda states: f"[{', '.join(states)}]"
    )

    # Save to CSV using semicolon as delimiter to avoid conflict with commas in the field
    output_file = "data/SignificantNamesByState.csv"
    significant_names.to_csv(output_file, index=False, sep=';', quoting=csv.QUOTE_NONE, escapechar='\\')

    print(f"Significant names saved to {output_file}")
    print(significant_names)

    # Step 4: Cluster states based on similar appeal across names using K-Modes
    # Adjust n_clusters as desired
    cluster_states(output_file, n_clusters=3)


if __name__ == "__main__":
    main()
