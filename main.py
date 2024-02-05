

import numpy as np
import pandas as pd #creating and manipulating dataframes
from matplotlib import ticker
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.pyplot as plt #visualsimport seaborn as sns #visuals
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import metrics
import seaborn as sns


data = pd.read_csv("covid/us_records_subset.csv")
data_cleaned = data.dropna(subset=data.select_dtypes(include=[int, float]).columns)

data.columns = [
    'date', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_cases_per_million',
    'total_deaths_per_million', 'icu_patients', 'hosp_patients', 'weekly_hosp_admissions',
    'daily_case_change_rate', 'daily_death_change_rate', 'hospitalization_rate', 'icu_rate',
    'case_fatality_rate', '7day_avg_new_cases', '7day_avg_new_deaths', 'hospitalization_need',
    'icu_requirement'
]


# Classification
le = LabelEncoder()
y_encoded_cleaned = le.fit_transform(data_cleaned['hospitalization_need'])
X_cleaned = data_cleaned.select_dtypes(include=[int, float])
X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(
    X_cleaned, y_encoded_cleaned, test_size=0.2, random_state=42
)
# Standardize the features by removing the mean and scaling to unit variance
scaler = StandardScaler()

# Transform the training data
X_train_scaled_cleaned = scaler.fit_transform(X_train_cleaned)

# Transform the test data using the same scaler
X_test_scaled_cleaned = scaler.transform(X_test_cleaned)

# Create a Decision Tree Classifier
clf_tree = DecisionTreeClassifier(random_state=42)

# Create a Random Forest Classifier
clf_forest = RandomForestClassifier(random_state=42)

# Evaluate the Decision Tree Classifier using cross-validation
scores_tree = cross_val_score(clf_tree, X_train_scaled_cleaned, y_train_cleaned, cv=5)

# Evaluate the Random Forest Classifier using cross-validation
scores_forest = cross_val_score(clf_forest, X_train_scaled_cleaned, y_train_cleaned, cv=5)

# Calculate the mean and standard deviation of the cross-validation scores for the Decision Tree
mean_score_tree = np.mean(scores_tree)
std_score_tree = np.std(scores_tree)

# Calculate the mean and standard deviation of the cross-validation scores for the Random Forest
mean_score_forest = np.mean(scores_forest)
std_score_forest = np.std(scores_forest)

# Fit the Random Forest Classifier to the training data
clf_forest.fit(X_train_scaled_cleaned, y_train_cleaned)

# Extract feature importances from the trained Random Forest model
feature_importance = clf_forest.feature_importances_

# Get the names of the features from the original DataFrame
feature_names = X_train_cleaned.columns

# Create a dictionary mapping feature names to their importance scores
feature_importance_dict = dict(zip(feature_names, feature_importance))

# Print the results
print("Decision Tree - Mean Accuracy:", mean_score_tree, ", Standard Deviation:", std_score_tree)
print("Random Forest - Mean Accuracy:", mean_score_forest, ", Standard Deviation:", std_score_forest)
print("Feature Importance from Random Forest:", feature_importance_dict)



# Exploratory Data Analysis
data['date'] = pd.to_datetime(data['date'])

plt.figure(figsize=(12, 6))

sns.lineplot(x='date', y='total_cases', data=data, label='Total Cases')
sns.lineplot(x='date', y='total_deaths', data=data, label='Total Deaths')
sns.lineplot(x='date', y='hospitalization_rate', data=data, label='Hospitalization Rate')

plt.title('COVID-19 Trends')
plt.xlabel('Date')
plt.ylabel('Count/Rate')
plt.legend()
plt.grid(True)
plt.show()

selected_columns = [
    'total_cases', 'total_deaths', 'hospitalization_rate'
]

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_cleaned[selected_columns])

kmeans = KMeans(n_clusters=3, random_state=42)
data_cleaned['cluster'] = kmeans.fit_predict(data_scaled)

cluster_features = pd.DataFrame(data_scaled, columns=selected_columns)
cluster_features['cluster'] = data_cleaned['cluster']

plt.figure(figsize=(16, 12))
for i, column in enumerate(selected_columns):
    plt.subplot(2, 3, i + 1)  
    sns.boxplot(x='cluster', y=column, data=cluster_features)
    plt.title(f'Box Plot for {column}')

plt.tight_layout()
plt.show()

# Select the relevant columns
selected_columns = ['total_cases', 'total_deaths', 'hospitalization_rate']

# Create a new DataFrame with only the selected columns
selected_data = data[selected_columns]

# Calculate the correlation matrix
correlation_matrix = selected_data.corr()

# Print the correlation matrix
print(correlation_matrix)



# Regression task
data = data.dropna()
data.shape
print(data)

# Removing categorical variables
data = data.iloc[:, :16]


# Seeing correlation btwn each of our variables and our target variable 'ICU_patients'
print(data.corr()['icu_patients'])

print(abs(data.corr()['icu_patients']).sort_values(ascending = False))
print(abs(data.corr()['icu_patients']).sort_values(ascending = False).index)

# Top 4 most correlated variables in relation to target variable 'icu_patients' (not including icu_patients_)

X_cols = list(abs(data.corr()['icu_patients']).sort_values(ascending = False).index[1:5])
print("The top 4 most correlated variables in relation to target variable icu_patients are \n", X_cols)

# Fitting the Regression Model
X = data[X_cols]
y = data['icu_patients']

# Using the last 20% of the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .20, random_state= 29)

# Printing Training Size for X and y
print("Training size for X", X_train.shape)
print("Training size for y", y_train.shape)

# Printing Test Size for X and y
print("Test size for X", X_test.shape)
print("Test size for y", y_test.shape)

# Fitting a Linear Regression Model to training data
lr = LinearRegression()
lr.fit(X_train,y_train)

# Using the Trained Model to Predict the Values for X_test
y_pred = lr.predict(X_test)

# Displaying the first few values for validaiton
print(y_pred[:5])

# Printing Coefficients, Intercepts, and Accuracy Scores
print('Coefficients: \n', lr.coef_)
print('Intercept: \n', lr.intercept_)

# Checking Accuracy
# Appending the actual and the predicted into the same dataframe
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred })

results.Predicted = results.Predicted.round(0).astype(int)
results.Actual = results.Actual.round(0).astype(int)
results.head()

 # Printing Mean Square Error
print('Residual sum of squares: %.2f'% np.mean((lr.predict(X_test) - y_test) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lr.score(X_test, y_test))

# R2 score
accuracy = metrics.r2_score(y_test, y_pred)*100
print('Accuracy of the model is %.2f' %accuracy)

# Plot of Actual ICU Patients vs Predicted ICU Patients
plt.scatter(y_test, y_pred)
plt.xlabel('ICU Patients')
plt.ylabel('Predicted ICU patients')
plt.show()



#Outliers
data['date'] = pd.to_datetime(data['date'])

#Calculate IQR for new cases
Q1_cases = data['new_cases'].quantile(0.25)
Q3_cases = data['new_cases'].quantile(0.75)
IQR_cases = Q3_cases - Q1_cases

#Define upper and lower bounds
lower_bound_cases = Q1_cases - 1.5 * IQR_cases
upper_bound_cases = Q3_cases + 1.5 * IQR_cases

#Identify outlier
outliers_cases = (data['new_cases'] > upper_bound_cases)

#Print IQR and total number of outliers 
total_outliers = outliers_cases.sum()
print("Interquartile Range for new cases", IQR_cases)
print("Total number of outliers in new cases:", total_outliers)

#Scatter plot for new_cases with date labels in months
plt.figure(figsize=(12, 8))
sns.scatterplot(x=data['date'], y=data['new_cases'], color='blue', label='Cases')
sns.scatterplot(x=data.loc[outliers_cases, 'date'], y=data.loc[outliers_cases, 'new_cases'], color='red', label='Outliers')
locator = MonthLocator() 
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
plt.title('Scatter Plot for New Cases with Date Labels (in Months)')
plt.xlabel('Date (in Months)')
plt.ylabel('New Cases')
plt.xticks(rotation=45, ha='right')  
plt.legend()
plt.show()

