import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# Part 1: Dataset Processing
# -------------------------

def label_length(name, q1, q2):
    """Label the name length as Short, Medium, or Long based on quantile thresholds.

    Parameters:
      name (str): The name.
      q1 (int): Lower threshold (33rd percentile).
      q2 (int): Upper threshold (66th percentile).

    Returns:
      str: 'Short' if name length <= q1, 'Medium' if between q1 and q2, 'Long' otherwise.
    """
    l = len(name)
    if l <= q1:
        return "Short"
    elif l <= q2:
        return "Medium"
    else:
        return "Long"


def compute_unisex(df):
    """
    Create a mapping from name to a boolean that is True if the name appears with both genders.
    """
    gender_counts = df.groupby('Name')['Gender'].nunique()
    return gender_counts >= 2


def compute_rarity(df):
    """
    For each gender, compute total counts per name and label the name as
    'Common' if it is in the top 10% for that gender, otherwise 'Rare'.
    Returns a DataFrame with columns: Name, rarity, and Gender.
    """
    rarity_list = []
    for gender in df['Gender'].unique():
        sub = df[df['Gender'] == gender].groupby('Name', as_index=False)['Count'].sum()
        top_n = int(np.ceil(len(sub) * 0.1))
        sub_sorted = sub.sort_values(by='Count', ascending=False).reset_index(drop=True)
        sub_sorted['rarity'] = "Rare"  # default label
        sub_sorted.loc[:top_n - 1, 'rarity'] = "Common"
        rarity_list.append(sub_sorted[['Name', 'rarity']].assign(Gender=gender))
    rarity_df = pd.concat(rarity_list)
    return rarity_df


def compute_trend(df):
    """
    Compute popularity trend for each (Name, Gender) pair.
    Aggregates counts per year, takes the last 10 years (if available),
    and checks if the counts are strictly increasing or strictly decreasing.
    Returns a dictionary with keys (Name, Gender) and trend as value.
    """
    trends = {}
    for (name, gender), group in df.groupby(['Name', 'Gender']):
        group_sorted = group.sort_values('Year')
        if group_sorted['Year'].nunique() >= 10:
            last10 = group_sorted.groupby('Year')['Count'].sum().tail(10).values
            if np.all(np.diff(last10) > 0):
                trend = "Increasing"
            elif np.all(np.diff(last10) < 0):
                trend = "Decreasing"
            else:
                trend = "Varied"
        else:
            trend = "Varied"
        trends[(name, gender)] = trend
    return trends


def process_dataset(df):
    """
    Process the baby names dataset and produce a new DataFrame with the following columns:
      - name: baby name
      - gender: baby gender
      - length: Short, Medium, or Long based on the name's length (using quantile thresholds)
      - unisex: Boolean flag (True if the name appears with both genders)
      - rarity: 'Common' if the name is in the top 10% for that gender, 'Rare' otherwise
      - popularity_trend: 'Increasing', 'Decreasing', or 'Varied'
    """
    df_new = df.rename(columns={'Name': 'name', 'Gender': 'gender'})

    # Compute quantiles for name lengths using the original Name column
    name_lengths = df['Name'].str.len()
    quantiles = name_lengths.quantile([0.33, 0.66])
    q33 = int(quantiles.iloc[0])
    q66 = int(quantiles.iloc[1])

    # Compute name length category using quantile-based thresholds
    df_new['length'] = df_new['name'].apply(lambda name: label_length(name, q33, q66))

    # Compute unisex flag mapping and merge into df_new by matching on name
    unisex_map = compute_unisex(df)
    df_new['unisex'] = df_new['name'].map(unisex_map)

    # Compute rarity per name and gender, then merge the rarity info
    rarity_df = compute_rarity(df)
    # Rename columns in rarity_df to lowercase for consistency
    rarity_df = rarity_df.rename(columns={'Name': 'name', 'Gender': 'gender'})
    df_new = df_new.merge(rarity_df, on=['name', 'gender'], how='left')

    # Compute popularity trend per name and gender
    trend_map = compute_trend(df)
    df_new['popularity_trend'] = df_new.apply(lambda row: trend_map.get((row['name'], row['gender'])), axis=1)

    return df_new


# -------------------------
# Part 2: Name Recommendation
# -------------------------

def suggest_names_based_on_preferences(processed_df, preferences, top_n=5):
    """
    Suggest names by comparing user preferences with each name's features
    using cosine similarity.

    processed_df: DataFrame with columns:
      ['name', 'gender', 'length', 'rarity', 'popularity_trend', 'unisex']
    user_preferences: Dictionary with keys:
      'length', 'unisex', 'gender', 'rarity', 'popularity_trend'
    top_x: Number of name suggestions to return.
    """
    # Prepare the feature matrix: ensure unisex is numeric
    features = processed_df[['length', 'gender', 'rarity', 'popularity_trend', 'unisex']].copy()
    features['unisex'] = features['unisex'].astype(int)

    # One-hot encode categorical features
    features_enc = pd.get_dummies(features, columns=['length', 'gender', 'rarity', 'popularity_trend'])

    # Create a DataFrame for the user's preferences
    user_df = pd.DataFrame([preferences])
    user_df['unisex'] = int(user_df['unisex'])
    user_enc = pd.get_dummies(user_df, columns=['length', 'gender', 'rarity', 'popularity_trend'])

    # Align user encoding with the full feature set (fill missing columns with 0)
    user_enc = user_enc.reindex(columns=features_enc.columns, fill_value=0)

    # Calculate cosine similarity between the user vector and all name feature vectors
    sim_scores = cosine_similarity(user_enc, features_enc)[0]

    # Add similarity scores to the processed DataFrame
    processed_df = processed_df.copy()
    processed_df['similarity'] = sim_scores

    # Sort names by similarity score (highest first) and return unique name suggestions
    suggestions = processed_df.sort_values(by='similarity', ascending=False)['name'].unique()[:top_n]
    return suggestions

if __name__ == "__main__":
    # Set input and output file names
    input_file = f'data/NationalNames.csv'
    processed_file = f'data/NameSuggestionsDataset.csv'

    # Read and process the original dataset
    df_original = pd.read_csv(input_file)
    processed_dataset = process_dataset(df_original)

    # Save the processed dataset to a CSV file
    processed_dataset.to_csv(processed_file, index=False)
    print("Processed dataset saved to:", processed_file)

    user_preferences = {
        "length": "Long",  # Options: Short, Medium, Long
        "unisex": True,  # Boolean: True for unisex names, False otherwise
        "gender": "F",  # Expected baby's gender: F or M
        "rarity": "Rare",  # Options: Rare or Common
        "popularity_trend": "Decreasing"  # Options: Increasing, Decreasing, Varied
    }

    # Get the top n name suggestions based on user preferences
    name_suggestions = suggest_names_based_on_preferences(pd.read_csv(f'data/NameSuggestionsDataset.csv'), user_preferences, top_n=10)
    print("Top name suggestions based on preferences:", name_suggestions)
