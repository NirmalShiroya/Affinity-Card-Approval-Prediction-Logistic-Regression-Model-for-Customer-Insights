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



"""# **Build Logistics Regression Model for 1000 Random Customers (Data Mining)**"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Step 1: Select 1000 random records from the DataFrame
random_sample = df.sample(n=1000)

# Step 2: Define the feature set (X) and the target variable (y)
# 'AFFINITY_CARD' is the target variable indicating whether a customer has an affinity card
target_variable = 'AFFINITY_CARD'
X = random_sample.drop(target_variable, axis=1)  # Features (excluding the target variable)
y = random_sample[target_variable]  # Target variable

# Step 3: Convert categorical features to numeric using dummy variables
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical variables, dropping the first category to avoid multicollinearity

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Standardize the features to ensure they are on the same scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Build a logistic regression model
logistic_regression_model = LogisticRegression(solver='liblinear', random_state=100)
logistic_regression_model.fit(X_train_scaled, y_train)  # Fit the model to the training data

# Step 7: Make predictions on the test set
y_pred = logistic_regression_model.predict(X_test_scaled)

# Step 8: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(classification)

# Step 9: Save the remaining data that was not used in the analysis
# Determine which records were not in the random sample
remaining_data = df.drop(random_sample.index)  # Drop the sampled records from the original data
remaining_data.to_csv("remaining_data.csv", index=False)  # Save to CSV

print("Remaining data saved to 'remaining_data.csv'.")

"""# **Build Prediction Application based on Logistic Regression Model**"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
import ipywidgets as widgets
from IPython.display import display
import io

# Load the dataset and preprocess it
data = pd.read_csv("/content/Marketing Campaign data.csv")  # Update with your file path
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset
data = data[['AGE', 'CUST_GENDER', 'CUST_MARITAL_STATUS', 'CUST_INCOME_LEVEL', 'COUNTRY_NAME', 'EDUCATION', 'OCCUPATION', 'HOUSEHOLD_SIZE', 'YRS_RESIDENCE',
             'BULK_PACK_DISKETTES', 'FLAT_PANEL_MONITOR', 'HOME_THEATER_PACKAGE', 'BOOKKEEPING_APPLICATION', 'Y_BOX_GAMES', 'OS_DOC_SET_KANJI', 'AFFINITY_CARD']]

# Encoding dictionaries for custom variables

gender_levels = {'F': 0, 'M': 1}  # Updated to match input data
education_levels = {'Presch.': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6, '12th': 7,
                    'HS-grad': 8, 'Assoc-A': 13, 'Assoc-V': 14, 'Profsc': 15, 'Bach.': 10, 'Masters': 11, 'PhD': 12, '< Bach.': 9}
household_levels = {'1': 0, '2': 1, '3': 2, '4-5': 3, '6-8': 4, '9+': 5}
income_levels = {'J: 190,000 - 249,999': 3, 'I: 170,000 - 189,999': 3, 'H: 150,000 - 169,999': 3,
                 'B: 30,000 - 49,999': 1, 'K: 250,000 - 299,999': 3, 'L: 300,000 and above': 3,
                 'G: 130,000 - 149,999': 3, 'C: 50,000 - 69,999': 1, 'E: 90,000 - 109,999': 2,
                 'D: 70,000 - 89,999': 2, 'F: 110,000 - 129,999': 2, 'A: Below 30,000': 1}
country_names = {'United States of America': 1, 'Brazil': 2, 'Argentina': 3, 'Germany': 4, 'Italy': 5, 'New Zealand': 6, 'Australia': 7,
                 'Poland': 8, 'Saudi Arabia': 9, 'Denmark': 10, 'Japan': 11, 'China': 12, 'Canada': 13, 'United Kingdom': 14, 'Singapore': 15,
                 'South Africa': 16, 'France': 17, 'Turkey': 18, 'Spain': 19}
marital_statuses = {'NeverM': 1, 'Married': 2, 'Divorc.': 3, 'Mabsent': 4, 'Separ.': 5, 'Widowed': 6, 'Mar-AF': 7}
occupations = {'Prof.': 1, 'Sales': 2, 'Cleric.': 3, 'Exec.': 4, 'Other': 5, 'Farming': 6, 'Transp.': 7, 'Machine': 8, 'Crafts': 9, 'Handler': 10,
               'Protec.': 11, 'TechSup': 12, 'House-s': 13, 'Armed-F': 14}

# Define entry widgets for each variable
widgets_list = []
for column in data.columns[:-1]:  # Exclude 'CUST_ID'
    if column == 'AGE':
        widget = widgets.FloatSlider(min=18, max=100, step=1, value=30, description='Age:')
    elif column == 'CUST_GENDER':
        widget = widgets.Dropdown(options=list(gender_levels.keys()), value=list(gender_levels.keys())[0], description='Gender:')
    elif column == 'CUST_MARITAL_STATUS':
        widget = widgets.Dropdown(options=list(marital_statuses.keys()), value=list(marital_statuses.keys())[0], description='Marital Status:')
    elif column == 'CUST_INCOME_LEVEL':
        widget = widgets.Dropdown(options=list(income_levels.keys()), value=list(income_levels.keys())[0], description='Income Level:')
    elif column == 'EDUCATION':
        widget = widgets.Dropdown(options=list(education_levels.keys()), value=list(education_levels.keys())[0], description='Education:')
    elif column == 'OCCUPATION':
        widget = widgets.Dropdown(options=list(occupations.keys()), value=list(occupations.keys())[0], description='Occupation:')
    elif column == 'HOUSEHOLD_SIZE':
        widget = widgets.FloatSlider(min=min(household_levels.values()), max=max(household_levels.values()), step=1,
                                      value=round(sum(household_levels.values()) / len(household_levels)), description='Household Size:')
    elif column == 'YRS_RESIDENCE':
        widget = widgets.FloatSlider(min=0, max=14, step=1, value=7, description='Years of Residence:')
    elif column in ['BULK_PACK_DISKETTES', 'FLAT_PANEL_MONITOR', 'HOME_THEATER_PACKAGE', 'BOOKKEEPING_APPLICATION', 'Y_BOX_GAMES', 'OS_DOC_SET_KANJI']:
        widget = widgets.Dropdown(options=['0', '1'], value='0', description=column.replace('_', ' ').title())
    elif column == 'COUNTRY_NAME':
        widget = widgets.Dropdown(options=list(country_names.keys()), value=list(country_names.keys())[0], description='Country Name:')
    else:
        widget = None
    if widget:
        widgets_list.append(widget)

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
X_train = train_data.drop('AFFINITY_CARD', axis=1)
y_train = train_data['AFFINITY_CARD']
X_test = test_data.drop('AFFINITY_CARD', axis=1)
y_test = test_data['AFFINITY_CARD']

# Apply ordinal encoding to categorical features using combined dataset
combined_data = pd.concat([X_train, X_test], axis=0)
encoder = OrdinalEncoder()
combined_data_encoded = pd.DataFrame(encoder.fit_transform(combined_data), columns=combined_data.columns)

# Split the combined dataset back into training and testing sets
X_train_encoded = combined_data_encoded[:len(X_train)]
X_test_encoded = combined_data_encoded[len(X_train):]

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_encoded, y_train)

def predict(input_data):
    DEFAULT_VALUE = 0
    input_data_encoded = []
    for i, value in enumerate(input_data):
        if isinstance(value, str):
            if i == 0:  # AGE
                input_data_encoded.append(float(value))
            elif i == 1:  # CUST_GENDER
                input_data_encoded.append(gender_levels[value])
            elif i == 2:  # CUST_MARITAL_STATUS
                input_data_encoded.append(marital_statuses[value])
            elif i == 3:  # CUST_INCOME_LEVEL
                input_data_encoded.append(income_levels[value])
            elif i == 4:  # COUNTRY_NAME
                input_data_encoded.append(country_names[value])
            elif i == 5:  # EDUCATION
                input_data_encoded.append(education_levels[value])
            elif i == 6:  # OCCUPATION
                input_data_encoded.append(occupations[value])
            elif i == 9 or i == 10 or i == 11 or i == 12 or i == 13 or i == 14:  # BULK_PACK_DISKETTES, FLAT_PANEL_MONITOR, HOME_THEATER_PACKAGE, BOOKKEEPING_APPLICATION, Y_BOX_GAMES, OS_DOC_SET_KANJI
                input_data_encoded.append(float(value))
            else:
                input_data_encoded.append(DEFAULT_VALUE)
        else:
            input_data_encoded.append(float(value))

    input_data_encoded = [input_data_encoded]
    input_data_encoded = np.array(input_data_encoded, dtype=np.float64)

    prediction = model.predict(input_data_encoded)
    probability = model.predict_proba(input_data_encoded)

    return prediction[0], probability[0]

def predict_uploaded_data(uploaded_data):
    # Drop any extra columns not present in the original dataset, ignoring errors
    uploaded_data = uploaded_data.drop(['CUST_ID','COMMENTS', 'AFFINITY_CARD', 'PRINTER_SUPPLIES'], axis=1, errors='ignore')

    # Take random 100 records from the uploaded data
    uploaded_data_sample = uploaded_data.sample(n=100, random_state=42)

    # Reorder columns to match the order of columns in the original dataset
    uploaded_data_sample = uploaded_data_sample[data.columns[:-1]]

    # Apply the same preprocessing steps as done for training data
    # Assuming the structure of uploaded data is similar to the original data
    uploaded_data_encoded = pd.DataFrame(encoder.transform(uploaded_data_sample), columns=uploaded_data_sample.columns)

    # Predict using the uploaded data
    for index, row in uploaded_data_encoded.iterrows():
        prediction, probability = predict(row)
        print(f"The Customer is eligible for AFFINITY_CARD (1/0)(Yes/No): {prediction}")
        print(f"Probability of being class 0: {probability[0]:.2f}")
        print(f"Probability of being class 1: {probability[1]:.2f}")


# Define function to handle file upload
def handle_file_upload(change):
    uploaded_filename = next(iter(file_upload.value))
    uploaded_file = file_upload.value[uploaded_filename]
    uploaded_data = pd.read_csv(io.BytesIO(uploaded_file['content']))
    # Predict using uploaded data
    predict_uploaded_data(uploaded_data)

def on_predict_button_clicked(b):
    input_data = [widget.value for widget in widgets_list]
    # Include CUST_ID automatically
    #input_data.insert(0, data['CUST_ID'].max() + 1)  # Assuming CUST_ID starts from 1 and increments by 1
    print(f"Input Data: {input_data}")
    prediction, probability = predict(input_data)
    print(f"The Customer is eligible for AFFINITY_CARD (1/0)(Yes/No): {prediction}")
    print(f"Probability of being class 0: {probability[0]:.2f}")
    print(f"Probability of being class 1: {probability[1]:.2f}")

# Display widgets and prediction button
for widget in widgets_list:
    display(widget)
predict_button = widgets.Button(description="Predict")
display(predict_button)

# Register the event handler
predict_button.on_click(on_predict_button_clicked)

# Display file upload widget
file_upload = widgets.FileUpload(accept='.csv', multiple=False)
file_upload.observe(handle_file_upload, names='value')
display(file_upload)

# Test the accuracy of the application
accuracy = model.score(X_test_encoded, y_test)
print(f"Accuracy of the application on the test data: {accuracy}")

"""# **Test Application based on logistic regression model using 100 random records from remaining datasets**"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import ipywidgets as widgets
import io

# List of columns used in the model
required_columns = ['AGE', 'CUST_GENDER', 'CUST_MARITAL_STATUS', 'CUST_INCOME_LEVEL', 'COUNTRY_NAME', 'EDUCATION', 'OCCUPATION', 'HOUSEHOLD_SIZE', 'YRS_RESIDENCE',
                    'BULK_PACK_DISKETTES', 'FLAT_PANEL_MONITOR', 'HOME_THEATER_PACKAGE', 'BOOKKEEPING_APPLICATION', 'Y_BOX_GAMES', 'OS_DOC_SET_KANJI']

# File upload widget
file_upload = widgets.FileUpload(accept='.csv', multiple=False)

# Function to handle file upload and process data
def handle_file_upload(change):
    # Get the uploaded file content
    uploaded_filename = list(file_upload.value.keys())[0]
    uploaded_file = file_upload.value[uploaded_filename]
    uploaded_data = pd.read_csv(io.BytesIO(uploaded_file['content']))

    # Select only the required columns
    # This ensures we're ignoring extra columns that aren't part of the model's input features
    uploaded_data_filtered = uploaded_data[required_columns]

    # Take a random sample of 100 customer records
    sample_data = uploaded_data_filtered.sample(n=100, random_state=42)

    # Apply encoding and make predictions
    sample_data_encoded = encoder.transform(sample_data)

    # Predict eligibility for affinity card
    for index, row in enumerate(sample_data_encoded):
        prediction, probability = predict(row)  # Assuming the predict function is defined
        print(f"Record {index + 1}: Eligible for Affinity Card: {prediction}")
        print(f"Probability of being class 0: {probability[0]:.2f}")
        print(f"Probability of being class 1: {probability[1]:.2f}")

# Attach the event handler to the file upload widget
file_upload.observe(handle_file_upload, names='value')

# Display the file upload widget
display(file_upload)
