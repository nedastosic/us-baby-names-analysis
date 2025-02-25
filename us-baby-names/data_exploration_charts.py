import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """
    Load datasets for state and national birth records.

    Returns:
      tuple: A tuple containing two pandas DataFrames:
             - df_states: DataFrame with state-level birth records.
             - df_national: DataFrame with national-level birth records.
    """
    df_states = pd.read_csv("data/StateNames.csv")
    df_national = pd.read_csv("data/NationalNames.csv")
    return df_states, df_national


def plot_all_states_trends(df_states, top_x=5):
    """
    Plot birth trends for all states over time and annotate the top states based on the final year's totals.

    This function aggregates the total count per state for each year, identifies the top `top_x` states
    in the final year, and plots the trends for all states. The top states are annotated with their names.

    Parameters:
      df_states (DataFrame): DataFrame containing state-level birth records.
      top_x (int): Number of top states to annotate based on the final year's totals.
    """
    # Aggregate total count per state over time
    state_trends = df_states.groupby(["Year", "State"])['Count'].sum().unstack()

    # Use a distinct color palette
    colors = sns.color_palette("tab20", n_colors=len(state_trends.columns))

    # Determine the final year in the dataset
    final_year = state_trends.index.max()

    # Identify the top_x states based on the final year's totals
    top_states = state_trends.loc[final_year].nlargest(top_x)
    print("Top", top_x, "states in", final_year, ":", top_states)

    # Plot trends for all states on one chart
    plt.figure(figsize=(14, 8))
    for state, color in zip(state_trends.columns, colors):
        plt.plot(state_trends.index, state_trends[state], label=state, alpha=0.8, color=color)

    # Annotate the plot with the names of the top_x states near their final data point
    for state in top_states.index:
        final_count = state_trends.loc[final_year, state]
        plt.text(final_year + 0.5, final_count, state, fontsize=6, verticalalignment='center')

    plt.xlabel("Year")
    plt.ylabel("Total Births")
    plt.title("Birth Trends Across All States")
    plt.legend(loc='upper right', fontsize='small', ncol=3, frameon=False, bbox_to_anchor=(1.25, 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_last_10_years_trends(df_states, top_x=5):
    """
    Plot birth trends for all states for the last 10 years and annotate the top states in the final year of that period.

    This function filters the state-level dataset for the last 10 years, aggregates total births per state,
    identifies the top `top_x` states in the final year of this period, and plots the trends for all states
    in the last 10 years with annotations for the top states.

    Parameters:
      df_states (DataFrame): DataFrame containing state-level birth records.
      top_x (int): Number of top states to annotate based on the final year's totals in the last 10 years.
    """
    # Aggregate total count per state over time
    state_trends = df_states.groupby(["Year", "State"])['Count'].sum().unstack()

    # Filter for the last 10 years only
    last_year = state_trends.index.max()
    first_year = last_year - 9
    state_trends_last10 = state_trends.loc[first_year:last_year]

    # Use a distinct color palette
    colors = sns.color_palette("tab20", n_colors=len(state_trends_last10.columns))

    # Identify the top_x states in the last year of the last 10 years
    top_states = state_trends_last10.loc[last_year].nlargest(top_x)
    print("Top", top_x, "states in", last_year, "over the last 10 years:", top_states)

    # Plot trends for all states in the last 10 years
    plt.figure(figsize=(14, 8))
    for state, color in zip(state_trends_last10.columns, colors):
        plt.plot(state_trends_last10.index, state_trends_last10[state], label=state, alpha=0.8, color=color)

    # Annotate the plot with the names of the top_x states near their final data point
    for state in top_states.index:
        final_count = state_trends_last10.loc[last_year, state]
        plt.text(last_year + 0.5, final_count, state, fontsize=10, verticalalignment='center')

    plt.xlabel("Year")
    plt.ylabel("Total Births")
    plt.title("Birth Trends Across All States (Last 10 Years)")
    plt.legend(loc='upper right', fontsize='small', ncol=3, frameon=False, bbox_to_anchor=(1.25, 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_gender_distribution(df_states):
    """
    Plot the total male and female births per state using a bar chart

    This function aggregates the total birth counts for each state by gender and visualizes the results
    using a non-stacked bar chart. Blue and pink colors are applied to represent male and female births distinctly.

    Parameters:
      df_states (DataFrame): DataFrame containing state-level birth records.
    """
    # Aggregate count of males and females per state
    gender_counts = df_states.groupby(["State", "Gender"])['Count'].sum().unstack()

    # Define pastel colors with gender mapping: light blue for males and light pink for females
    pastel_colors = {"M": "#87CEFA", "F": "#FFB6C1"}

    plt.figure(figsize=(12, 6))
    gender_counts.plot(kind='bar', stacked=False, figsize=(12, 6),
                       color=[pastel_colors[g] for g in gender_counts.columns])
    plt.xlabel("State")
    plt.ylabel("Total Births")
    plt.title("Total Male and Female Births per State")
    plt.legend(title="Gender", bbox_to_anchor=(1.25, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_top_names_over_time(df_national):
    """
    Plot trends for the top 10 female and top 10 male names over time using line plots.

    This function identifies the overall top 10 names for each gender based on total counts, filters the national dataset,
    and then plots the trends over time for these names with separate plots for female and male names.

    Parameters:
      df_national (DataFrame): DataFrame containing national-level birth records.
    """
    # Identify top 10 female and male names overall
    top_10_female = df_national[df_national['Gender'] == 'F'].groupby("Name")['Count'].sum().nlargest(10).index
    top_10_male = df_national[df_national['Gender'] == 'M'].groupby("Name")['Count'].sum().nlargest(10).index

    # Filter data for these names
    female_trends = df_national[(df_national['Name'].isin(top_10_female)) & (df_national['Gender'] == 'F')]
    male_trends = df_national[(df_national['Name'].isin(top_10_male)) & (df_national['Gender'] == 'M')]

    # Plot top 10 female names trends
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=female_trends, x="Year", y="Count", hue="Name", palette="tab10")
    plt.xlabel("Year")
    plt.ylabel("Total Births")
    plt.title("Top 10 Female Names Over Time")
    plt.legend(title="Name", bbox_to_anchor=(1.25, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Plot top 10 male names trends
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=male_trends, x="Year", y="Count", hue="Name", palette="tab10")
    plt.xlabel("Year")
    plt.ylabel("Total Births")
    plt.title("Top 10 Male Names Over Time")
    plt.legend(title="Name", bbox_to_anchor=(1.25, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_recent_names(df_national):
    """
    Plot trends for the top 10 female and top 10 male names in the last 10 years using line plots.

    This function filters the national dataset to include only the data from the last 10 years, determines
    the top 10 names for each gender in this period, and then visualizes the trends over time for these names.

    Parameters:
      df_national (DataFrame): DataFrame containing national-level birth records.
    """
    # Identify the last 10 years of data
    last_10_years = df_national[df_national['Year'] >= df_national['Year'].max() - 9]
    top_recent_female = last_10_years[last_10_years['Gender'] == 'F'].groupby("Name")['Count'].sum().nlargest(10).index
    top_recent_male = last_10_years[last_10_years['Gender'] == 'M'].groupby("Name")['Count'].sum().nlargest(10).index

    # Filter data for these names
    recent_female_trends = last_10_years[
        (last_10_years['Name'].isin(top_recent_female)) & (last_10_years['Gender'] == 'F')]
    recent_male_trends = last_10_years[(last_10_years['Name'].isin(top_recent_male)) & (last_10_years['Gender'] == 'M')]

    # Plot top 10 female names in the last 10 years
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=recent_female_trends, x="Year", y="Count", hue="Name", palette="tab10")
    plt.xlabel("Year")
    plt.ylabel("Total Births")
    plt.title("Top 10 Female Names in Last 10 Years")
    plt.legend(title="Name", bbox_to_anchor=(1.25, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Plot top 10 male names in the last 10 years
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=recent_male_trends, x="Year", y="Count", hue="Name", palette="tab10")
    plt.xlabel("Year")
    plt.ylabel("Total Births")
    plt.title("Top 10 Male Names in Last 10 Years")
    plt.legend(title="Name", bbox_to_anchor=(1.25, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def main():
    # Load the datasets
    df_states, df_national = load_data()

    # Plot state trends with annotated top states
    plot_all_states_trends(df_states, top_x=10)

    # Plot trends for the last 10 years only
    plot_last_10_years_trends(df_states, top_x=10)

    # Plot gender distribution per state
    plot_gender_distribution(df_states)

    # Plot trends for top names over time (nationally)
    plot_top_names_over_time(df_national)

    # Plot trends for recent top names (last 10 years)
    plot_recent_names(df_national)


if __name__ == "__main__":
    main()
