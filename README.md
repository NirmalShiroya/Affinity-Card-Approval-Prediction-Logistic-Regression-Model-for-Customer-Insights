# Affinity Card Approval Prediction: Logistic Regression Model for Customer Insights

## Project Description
This project involves developing a logistic regression machine learning model to predict the likelihood of customers being approved for an affinity card based on their demographic and purchasing data. The model provides insights into customer behavior and helps streamline the decision-making process for affinity card approvals.

---

## Key Features
- **Data Preprocessing:** Cleaned and prepared the raw dataset for analysis, handling missing values, encoding categorical variables, and scaling features.
- **Exploratory Data Analysis (EDA):** Generated metadata, histograms, and summary statistics to understand data distributions and relationships.
- **Correlation Analysis:** Identified the most influential variables affecting affinity card approvals.
- **Model Development:** Built and tested a logistic regression model for predicting customer eligibility.
- **Random Sampling:** Evaluated the model using a random sample of 1000 records from the dataset.
- **Prediction Application:** Developed an interactive application to predict customer eligibility based on user input or uploaded datasets.
- **Performance Metrics:** Measured model accuracy, classification report, and confusion matrix to assess its effectiveness.

---

## Workflow
1. **Loading and Understanding the Dataset:**
   - Data was loaded, and metadata was generated for better understanding.
   - Key preprocessing steps included removing irrelevant columns, handling missing values, and encoding categorical variables.

2. **Data Exploration:**
   - Generated histograms and bar charts for visualizing variable distributions.
   - Analyzed relationships between features and the target variable (Affinity Card approval).

3. **Model Development:**
   - Built a logistic regression model to predict affinity card approvals.
   - Trained and tested the model on an 80-20 split of the dataset.

4. **Prediction Application:**
   - Created an interactive tool to predict customer eligibility based on input features or an uploaded dataset.
   - Designed the app to provide probabilities for each class (approval or rejection).

5. **Evaluation:**
   - Assessed model performance with metrics like accuracy, confusion matrix, and classification report.
   - Verified the application’s functionality on random samples and unseen data.

---

## Tools and Technologies
- **Programming Language:** Python
- **Libraries Used:**
  - `pandas` and `numpy` for data manipulation
  - `matplotlib` and `seaborn` for data visualization
  - `scikit-learn` for model building and evaluation
  - `ipywidgets` for interactive application development
- **Environment:** Google Colab

---

## How to Use
1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   ```

2. **Run the Code:**
   - Open the Jupyter Notebook or Google Colab file.
   - Execute the cells in sequence to preprocess the data, build the model, and generate predictions.

3. **Interactive Application:**
   - Use the provided widgets to input customer details or upload a dataset for batch predictions.

4. **Model Output:**
   - The application will display whether a customer is eligible for the affinity card along with the probability scores for each class.

---

## Dataset
- **Source:** Marketing Campaign data.
- **Features Include:**
  - Demographics: Age, Gender, Marital Status, etc.
  - Socioeconomic: Income Level, Education, Occupation.
  - Behavioral: Years of Residence, Product Purchases.

---

## Results
- **Accuracy:** The model achieved a high accuracy score on the test data.
- **Insights:** Key factors influencing affinity card approval include income level, marital status, and purchasing behavior.

---

## Future Enhancements
- Integrate more advanced machine learning models for comparison.
- Expand the dataset for greater generalizability.
- Enhance the interactive application with visualization features.

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request to suggest changes or improvements.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact
For any queries or suggestions, please contact [Your Name] at [Your Email].


