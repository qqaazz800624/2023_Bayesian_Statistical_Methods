#%%
import pandas as pd

#%%
# Load the dataset
file_path = 'naes04.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset for inspection
data.head()


#%%

# Filtering out missing data for the two questions of interest
data_filtered = data.dropna(subset=['gayFavorStateMarriage', 'gayKnowSomeone'])

# Grouping by age and calculating the proportion of "Yes" responses for each age
grouped_data = data_filtered.groupby('age').agg(
    total_respondents=pd.NamedAgg(column="age", aggfunc="size"),
    yes_gayFavorStateMarriage=pd.NamedAgg(column="gayFavorStateMarriage", aggfunc=lambda x: (x == "Yes").sum()),
    yes_gayKnowSomeone=pd.NamedAgg(column="gayKnowSomeone", aggfunc=lambda x: (x == "Yes").sum())
)

# Calculating the proportions
grouped_data['prop_gayFavorStateMarriage'] = grouped_data['yes_gayFavorStateMarriage'] / grouped_data['total_respondents']
grouped_data['prop_gayKnowSomeone'] = grouped_data['yes_gayKnowSomeone'] / grouped_data['total_respondents']

# Displaying the processed data
grouped_data.head()


#%%
