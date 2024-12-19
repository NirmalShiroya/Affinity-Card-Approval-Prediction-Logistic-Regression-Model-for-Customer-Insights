# **Loading the Datasets**
"""

import pandas as pd

# Read the CSV file
df = pd.read_csv('/content/Marketing Campaign data.csv')

print(df.head())

"""# **Meta Data Table for Data Understanding**"""

import pandas as pd
from tabulate import tabulate

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('/content/Marketing Campaign data.csv')

# Function to limit the display of unique values
shortened_unique_values = lambda lst, max_disp=3, max_len=20: (", ".join(map(str, lst[:max_disp]))
+ (", ..." if len(lst) > max_disp else "")
if len(", ".join(map(str, lst[:max_disp]))) <= max_len else ", ".join(map(str, lst[:max_disp]))[:max_len - 3] + "...")

# Function to limit the length of the mode
truncated_mode = lambda mode_str, max_len=15: (mode_str[:max_len - 3]
                                               + "..." if mode_str and len(mode_str) > max_len else mode_str)

# Create metadata
metadata = []
for column in df.columns:
    data_type = df[column].dtype
    unique_values = shortened_unique_values(df[column].unique())
    max_value = df[column].max() if data_type in ['int64', 'float64'] else None
    min_value = df[column].min() if data_type in ['int64', 'float64'] else None
    mean = df[column].mean() if data_type in ['int64', 'float64'] else None
    std_dev = df[column].std() if data_type in ['int64', 'float64'] else None
    mode = truncated_mode(str(df[column].mode().values[0])) if len(df[column].mode()) > 0 else None
    histogram = True if data_type in ['int64', 'float64'] else False
    bar_chart = True if data_type not in ['int64', 'float64'] else False

    metadata.append((column, data_type, unique_values, max_value, min_value, mean, std_dev, mode, histogram, bar_chart))

# Create metadata DataFrame
metadata_df = pd.DataFrame(metadata, columns=['Variable Name', 'Data Type', 'Unique Values', 'Maximum', 'Minimum', 'Mean', 'Std. Deviation', 'Mode', 'Histogram', 'Bar Chart'])

# Save metadata to CSV
metadata_df.to_csv('metadata_table.csv', index=False)
print("Metadata table has been saved to metadata_table.csv")

# Display the metadata in a tabular format
metadata_table = tabulate(metadata_df, headers='keys', tablefmt='grid', showindex=False)
print("Metadata table for all attributes:")
print(metadata_table)

"""# **Histograms and Bar Charts for respective variables**"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Shorten x-axis labels
shorten_label = lambda label, max_len=5: label[:max_len - 1] + '...' if len(label) > max_len else label

# Initialize lists for histograms and bar charts
has_hist, has_bar = [], []

# Identify columns for histograms and bar charts
for col in df.columns:
    (has_hist if df[col].dtype in ['int64', 'float64'] else has_bar).append(col)

# Plot histograms
plt.figure(figsize=(15, 10))
for i, col in enumerate(has_hist):
    plt.subplot(math.ceil(len(has_hist) / 4), 4, i+1)
    sns.histplot(df[col], bins=10, kde=True)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Plot bar charts
plt.figure(figsize=(15, 10))
for i, col in enumerate(has_bar):
    plt.subplot(math.ceil(len(has_bar) / 4), 4, i+1)
    shortened_labels = [shorten_label(label) for label in df[col].value_counts().index]
    df[col].value_counts().rename(index=dict(zip(df[col].value_counts().index, shortened_labels))).plot(kind='bar')
    plt.title(f'Bar Chart of {col}')
    plt.xlabel('Category')
    plt.ylabel('Count')
plt.tight_layout()
plt.show()

"""# **Finding the Unique Values, Missing Values and Error in Datasets**"""

import pandas as pd

# Initialize lists to store results
columns = df.columns.tolist()
unique_values = [df[column].nunique() for column in columns]
total_missing_values = [(df[column].isnull().sum() + (df[column].astype(str).str.strip() == '').sum() +
                         df[column].astype(str).str.lower().str.strip().eq('unknown').sum()) for column in columns]
error_values = [(df[column] == "#VALUE!").sum() if df[column].dtype == 'object' else 0 for column in columns]

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Column': columns,
    'Unique Values': unique_values,
    'Total Missing Values': total_missing_values,
    'Error Values': error_values,
})

# Display the results DataFrame
print(results_df)

"""# **Remove Comments Columns as well as variable with ZERO influences**


"""

import pandas as pd

# Load your dataset
df = pd.read_csv('/content/Marketing Campaign data.csv')

# Check the unique values of the PRINTER_SUPPLIES column
print(df['PRINTER_SUPPLIES'].unique())

# Remove the PRINTER_SUPPLIES and COMMENTS columns
df = df.drop(['PRINTER_SUPPLIES', 'COMMENTS'], axis=1)

# Save the modified dataset
df.to_csv('modified_dataset.csv', index=False)

"""# **Finding Correlations with Target Variable "Affinity Card" and remove Comments Columns as well as variable with ZERO influences**"""

from sklearn.preprocessing import OrdinalEncoder

# Read the CSV file
df = pd.read_csv('/content/Marketing Campaign data.csv')

# Drop 'COMMENTS' column if it's not needed
df = df.drop(columns=['COMMENTS'])

# Apply Ordinal Encoding to categorical columns
ordinal_encoder = OrdinalEncoder()
df_encoded = df.copy()
df_encoded[df.select_dtypes(include=['object']).columns] = ordinal_encoder.fit_transform(df.select_dtypes(include=['object']))

# Calculate correlations with 'AFFINITY_CARD'
correlation = df_encoded.corr()['AFFINITY_CARD']

# Identify low-influential variables
low_influential = [key for key, value in correlation.items() if abs(value) == 0 or pd.isnull(value)]

# Remove low-influential variables from the DataFrame
df_filtered = df_encoded.drop(columns=low_influential)

# Display correlations with 'AFFINITY_CARD'
print("Correlation with ordinal encoding:")
print(correlation)

print("\nLow influential variables:")
print(low_influential)

# Display the names of the columns in the filtered DataFrame
print("\nFiltered DataFrame columns:")
print(df_filtered.columns)

print(df.shape)

"""# **Clean the records (remove records with missing values or errors if it’s less than 5%))**

1.   Remove Printer Supplies column and COMMENTS columns

2. Now Our DataFrame is 1500 X 17    


"""

# Calculate the percentage of missing or error data in each column
for column in df.columns:
    # Count missing values
    num_missing = df[column].isnull().sum()

    # Count error values (e.g., '#VALUE!')
    num_error = (df[column] == '#VALUE!').sum()

    # Calculate total number of records
    total_records = len(df)

    # Calculate the percentage of missing or error values
    percent_missing = (num_missing / total_records) * 100
    percent_error = (num_error / total_records) * 100

    # Remove records with missing values or errors if the percentage is less than 5%
    if percent_missing < 5:
        df = df.dropna(subset=[column])
    if percent_error < 5:
        df = df[df[column] != '#VALUE!']

# Print the cleaned DataFrame
print("Data frame after Cleaning missing values and error values from records")
print(df.shape)

"""# **Transforming some variables using pandas library and basic python code**"""

import pandas as pd

# Read the data into a pandas DataFrame
#df = pd.read_csv('/content/Marketing Campaign data.csv')

# a) Transforming CUST_GENDER into binary (F - 0, M - 1)
# Basic Python code
df_basic_gender = df.copy()
df_basic_gender['CUST_GENDER'] = df_basic_gender['CUST_GENDER'].map({'F': 0, 'M': 1})

# Pandas library method
df_pandas_gender = df.copy()
df_pandas_gender['CUST_GENDER'] = pd.factorize(df_pandas_gender['CUST_GENDER'])[0]

# Now, update the original DataFrame with the transformed columns using the pandas method for consistency
df.update(df_basic_gender[['CUST_GENDER']])

print(df.head())

# b) Transforming COUNTRY_NAME into ordinal numbers based on occurrence
# Basic Python code
country_counts = df['COUNTRY_NAME'].value_counts()
country_dict = {country: idx for idx, country in enumerate(country_counts.index)}
df_basic_country = df.copy()
df_basic_country['COUNTRY_NAME'] = df_basic_country['COUNTRY_NAME'].map(country_dict)

# Pandas library method
df_pandas_country = df.copy()
df_pandas_country['COUNTRY_NAME'] = df_pandas_country['COUNTRY_NAME'].astype('category').cat.codes

# Now, update the original DataFrame with the transformed columns using the pandas method for consistency
df.update(df_basic_country[['COUNTRY_NAME']])

print(df.head())

# c) Transforming CUST_INCOME_LEVEL into 3 ordinal levels (1: low, 2: middle, 3: high)
# Basic Python code
income_levels ={'J: 190,000 - 249,999': 3,'I: 170,000 - 189,999': 3, 'H: 150,000 - 169,999': 3,
 'B: 30,000 - 49,999': 1, 'K: 250,000 - 299,999': 3, 'L: 300,000 and above': 3,
 'G: 130,000 - 149,999': 3, 'C: 50,000 - 69,999': 1, 'E: 90,000 - 109,999': 2,
 'D: 70,000 - 89,999': 2, 'F: 110,000 - 129,999': 2, 'A: Below 30,000': 1}
df_basic_income = df.copy()
df_basic_income['CUST_INCOME_LEVEL'] = df_basic_income['CUST_INCOME_LEVEL'].map(income_levels)

# Pandas library method
df_pandas_income = df.copy()
df_pandas_income['CUST_INCOME_LEVEL'] = pd.factorize(df_pandas_income['CUST_INCOME_LEVEL'], sort=True)[0] + 1

# Now, update the original DataFrame with the transformed columns using the pandas method for consistency
df.update(df_basic_income[['CUST_INCOME_LEVEL']])

print(df.head())

# d) Transforming EDUCATION into ordinal numbers based on USA education levels
# Basic Python code
education_levels = {'Presch.': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6, '12th': 7,
                    'HS-grad': 8, 'Assoc-A': 9, 'Assoc-V': 10, 'Profsc': 11, 'Bach.': 12, 'Masters': 13, 'PhD': 14, '< Bach.': 15}
df_basic_education = df.copy()
df_basic_education['EDUCATION'] = df_basic_education['EDUCATION'].map(education_levels)

# Pandas library method
df_pandas_education = df.copy()
df_pandas_education['EDUCATION'] = pd.Categorical(
    df_pandas_education['EDUCATION'], categories=education_levels.keys(), ordered=True).codes

# Now, update the original DataFrame with the transformed columns using the pandas method for consistency
df.update(df_basic_education[['EDUCATION']])

print(df.head())

# e) Transforming HOUSEHOLD_SIZE into ordinal numbers based on the number of people
# Basic Python code
household_levels = {'1': 0, '2': 1, '3': 2, '4-5': 3, '6-8': 4, '9+': 5}
df_basic_household = df.copy()
df_basic_household['HOUSEHOLD_SIZE'] = df_basic_household['HOUSEHOLD_SIZE'].map(household_levels)

# Pandas library method
df_pandas_household = df.copy()
df_pandas_household['HOUSEHOLD_SIZE'] = pd.Categorical(
    df_pandas_household['HOUSEHOLD_SIZE'], categories=household_levels.keys(), ordered=True).codes

# Now, update the original DataFrame with the transformed columns using the pandas method for consistency
df.update(df_basic_household[['HOUSEHOLD_SIZE']])

print(df.head())

# Save the updated DataFrame to a new CSV file
df.to_csv('/content/Updated_Marketing_Campaign_data.csv', index=False)

print("Updated DataFrame has been saved to Updated_Marketing_Campaign_data.csv")

print(df.head)

"""# **Summary statistics of all variables for Data Analysis Without change the String value columns**"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Summary statistics for all columns
summary_stats_all = df.describe(include='all').transpose()

# Additional statistics: skewness and kurtosis for numeric columns
numeric_cols = df.select_dtypes(include=[np.number])
skewness = numeric_cols.apply(lambda x: skew(x.dropna()))
kurt = numeric_cols.apply(lambda x: kurtosis(x.dropna()))

# Fill NaN for numeric columns without skewness or kurtosis
skewness = skewness.fillna(np.nan)
kurt = kurt.fillna(np.nan)

# Concatenate the additional statistics to the summary statistics DataFrame for numeric columns
summary_stats_all['skewness'] = skewness
summary_stats_all['kurtosis'] = kurt

# Display the summary statistics for all columns
#print(summary_stats_all)

# Print summary statistics for all columns in tabular form, excluding top and unique values
print(summary_stats_all.to_string(index=True, columns=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']))

"""# **Summary statistics of all variables for Data Analysis With change the String value columns using Ordinal encoder**"""

from scipy.stats import skew, kurtosis
from sklearn.preprocessing import OrdinalEncoder
# Automatically find all string columns
string_columns = df.select_dtypes(include=[object]).columns

print("String Columns:")
print(string_columns)

# Ordinal encoding for string columns
encoder = OrdinalEncoder()
df_ordinal = df.copy()
df_ordinal[string_columns] = encoder.fit_transform(df[string_columns])

# Calculate summary statistics for all columns, including skewness and kurtosis
summary_stats_all = df_ordinal.describe(include='all').transpose()

# Skewness and kurtosis for numeric columns
numeric_cols = df_ordinal.select_dtypes(include=[np.number])
skewness = numeric_cols.apply(lambda x: skew(x.dropna()))
kurt = numeric_cols.apply(lambda x: kurtosis(x.dropna()))

# Add skewness and kurtosis to the summary statistics
summary_stats_all['skewness'] = skewness
summary_stats_all['kurtosis'] = kurt

# Summary statistics for all columns excluding 'top' and 'unique' values
summary_columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']
print("Summary Statistics:")
print(summary_stats_all.to_string(index=True, columns=summary_columns))

"""# **Histogram plot of a variable which allow a user to choose in runtime. (Data Exploration)**"""

import pandas as pd
import matplotlib.pyplot as plt
# Function to show a histogram plot of a given column
def show_histogram(column_name):
    plt.figure()
    df[column_name].hist(bins=10, edgecolor='black')  # Adjust the number of bins as needed
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()
# Main loop to interact with the user
while True:
    # List available numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    # Display available numeric columns to the user
    print("\nAvailable columns for histogram:")
    for idx, col in enumerate(numeric_columns, 1):
        print(f"{idx}. {col}")

    # Ask user to choose a column
    user_input = input("\nChoose a column number to plot a histogram (or 'exit' to quit): ")

    if user_input.lower() == 'exit':
        break  # Exit the loop and end the program

    # Validate user input
    try:
        column_index = int(user_input) - 1
        if column_index < 0 or column_index >= len(numeric_columns):
            raise ValueError("Invalid index")
        chosen_column = numeric_columns[column_index]
        # Show the histogram plot for the chosen column
        show_histogram(chosen_column)
    except ValueError:
        print("Invalid input. Please enter a valid column number or 'exit' to quit.")
