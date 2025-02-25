import pandas as pd
import jellyfish

# Load dataset and extract unique (Name, Gender) pairs
df = pd.read_csv('data/NationalNames.csv')
names_df = df[['Name', 'Gender']].drop_duplicates().reset_index(drop=True)
names = names_df['Name'].tolist()
genders = names_df['Gender'].tolist()


def hamming_distance(s1, s2):
    """Return the Hamming distance between two equal-length strings."""
    if len(s1) != len(s2):
        raise ValueError("Strings must be of equal length for Hamming distance.")
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))


# Precompute the Soundex codes for each unique name using jellyfish
name_soundex = {name: jellyfish.soundex(name) for name in set(names)}


def get_similar_names(query_name, top_n=10, gender=None):
    """
    Returns the top_n similar names to query_name based on phonetic similarity using Soundex.

    Parameters:
      query_name (str): The name to query.
      top_n (int): Number of similar names to return.
      gender (str, optional): Filter by gender ("M" or "F"). If None, no filter is applied.

    Returns:
      list: List of names sorted by phonetic similarity.
    """
    query_code = jellyfish.soundex(query_name)

    candidates = []
    for i, name in enumerate(names):
        if gender and genders[i] != gender:
            continue

        candidate_code = name_soundex.get(name)
        if candidate_code is None:
            continue

        # Compute Hamming distance between the query and candidate Soundex codes.
        distance = hamming_distance(query_code, candidate_code)
        candidates.append((name, distance))

    # Sort candidates by increasing Hamming distance
    candidates = sorted(candidates, key=lambda x: x[1])

    # Remove duplicate names (if any) and return top_n suggestions
    seen = set()
    result = []
    for name, _ in candidates:
        if name not in seen:
            seen.add(name)
            result.append(name)
        if len(result) >= top_n:
            break

    return result


if __name__ == "__main__":
    name = "James"
    desired_gender = "M"  # "M" or "F"; set to None to disable filtering
    similar_names = get_similar_names(name, top_n=10, gender=desired_gender)
    print(similar_names)
