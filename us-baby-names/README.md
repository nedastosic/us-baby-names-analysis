# US Baby Names Data Analysis

This project analyzes the [US Baby Names dataset](https://www.kaggle.com/datasets/kaggle/us-baby-names/data) from Kaggle. It features data exploration, state clustering, and two recommendation models—one based on user preferences and another based on phonetic similarity.

## Table of Contents

- [Overview](#overview)
- [Data Source](#data-source)
- [Project Structure](#project-structure)
- [Installation and Dependencies](#installation-and-dependencies)
- [Usage](#usage)
  - [Data Exploration](#data-exploration)
  - [States Clustering](#states-clustering)
  - [Name Recommendation Based on Preferences](#name-recommendation-based-on-preferences)
  - [Name Recommendation Based on Phonetic Similarity](#name-recommendation-based-on-phonetic-similarity)
- [License](#license)

## Overview

This project is designed to:
- **Explore and visualize** US baby names data by plotting trends, gender distributions, and top names over time
- **Cluster states** based on the popularity ratios of names compared to national data
- **Recommend names** by analyzing user-specified preferences (e.g., name length, unisex flag, rarity, popularity trend)
- **Suggest similar names** based on phonetic similarity using Soundex and Hamming distance

## Data Source

The dataset used in this project is available on [Kaggle](https://www.kaggle.com/datasets/kaggle/us-baby-names/data).

Download the files for national and states names and add them to the project. The CSV files should be placed in the 'data' folder and named 'NationalNames.csv' and 'StateNames.csv'.

## Project Structure

- **data_exploration_charts.py**
  Contains functions for loading the data and generating various charts to explore state and national baby name trends, including:
  - Overall state trends with annotated top states
  - Trends for the last 10 years
  - Gender distribution by state
  - Trends for top names over time (national level) and for recent years

- **states_clustering.py**
  Merges state and national datasets, computes state-to-national ratios, pivots the data, identifies names with significant state differences, and clusters states using K-Modes clustering. It also visualizes clusters using PCA.

- **names_preferencies_recommendation.py**
  Processes the baby names dataset to add features such as name length category, unisex flag, rarity, and popularity trend. It then implements a recommendation system that suggests names based on user preferences by calculating cosine similarity between feature vectors.

- **names_phonetic_recommendation.py**
  Uses phonetic similarity (via the Soundex algorithm and Hamming distance) to suggest similar names. It filters results by gender if specified.

## Installation and Dependencies

Ensure you have Python 3.6 or higher installed. Install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kmodes jellyfish
```

## Usage

### Data Exploration

Run the `data_exploration_charts.py` script to generate various charts:

```bash
python data_exploration_charts.py
```

This script will load state and national birth records, create trend plots (including overall trends and the last 10 years), visualize gender distributions, and plot top names over time.

### States Clustering

To perform clustering of states based on name popularity ratios and visualize the clusters using PCA, run:

```bash
python states_clustering.py
```

The script merges the datasets, pivots the data, identifies names with significant ratios, and clusters states using K-Modes. The results are printed to the console and visualized in a scatter plot.

### Name Recommendation Based on Preferences

This script processes the dataset to enrich it with features such as length category, unisex flag, rarity, and trend. It then recommends names based on a set of user preferences. Run:

```bash
python names_preferencies_recommendation.py
```

After processing the dataset, the script outputs a CSV file with the enriched data and prints the top name suggestions based on the default preferences. Modify the `user_preferences` dictionary in the script to experiment with different recommendation settings.

### Name Recommendation Based on Phonetic Similarity

For phonetic-based name suggestions, run the following script:

```bash
python names_phonetic_recommendation.py
```

This script computes the Soundex codes for names and suggests similar names (filtered by gender if specified) based on the Hamming distance between Soundex codes.
