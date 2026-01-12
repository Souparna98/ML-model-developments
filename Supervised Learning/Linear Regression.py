%run ../Includes/Utils

%md Display the catalog name, schema name, working directory, and dataset locations.

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

%md Load Dataset
/*
This dataset contains information about housing districts in California and aims to predict the median house value for California districts, based on various features.

While data cleaning and feature engineering is out of the scope of this demo, we will only map the ocean_proximity field.
*/

from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# read data from the feature store
table_name = f"{DA.catalog_name}.{DA.schema_name}.ca_housing"
feature_data_pd = fe.read_table(name=table_name).toPandas()
feature_data_pd = feature_data_pd.drop(columns=['unique_id'])

ocean_proximity_mapping = {
    'NEAR BAY': 1,
    '<1H OCEAN': 2,
    'INLAND': 3,
    'NEAR OCEAN': 4,
    'ISLAND': 5  
}

# Replace values in the DataFrame
feature_data_pd['ocean_proximity'] = feature_data_pd['ocean_proximity'].replace(ocean_proximity_mapping).astype(float)

# Print the updated DataFrame
feature_data_pd = feature_data_pd.fillna(0)

display(feature_data_pd)


%md Test/Train Split

-- Split the dataset into training and testing sets. This is essential for evaluating the performance of machine learning models.

from sklearn.model_selection import train_test_split

print(f"We have {feature_data_pd.shape[0]} records in our source dataset")

# split target variable into its own dataset
target_col = "median_house_value"
X_all = feature_data_pd.drop(labels=target_col, axis=1)
y_all = feature_data_pd[target_col]

# test / train split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.8, random_state=42)
print(f"We have {X_train.shape[0]} records in our training dataset")
print(f"We have {X_test.shape[0]} records in our test dataset")


%md Check for multi-collinearity

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Combine X and y into a single DataFrame for simplicity
data = pd.concat([X_train, y_train], axis=1)

# Calculate correlation matrix
corr = data.corr()

# display correlation matrix
pd.set_option('display.max_columns', 10)
print(corr)

# display correlation matrix visually

# Initialize figure
plt.figure(figsize=(8, 8))
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        # Determine the color based on positive or negative correlation
        color = 'blue' if corr.iloc[i, j] > 0 else 'red'

        # don't fill in circles on the diagonal
        fill = not( i == j )

        # Plot the circle with size corresponding to the absolute value of correlation
        plt.gca().add_patch(plt.Circle((j, i), 
                                       0.5 * np.abs(corr.iloc[i, j]), 
                                       color=color, 
                                       edgecolor=color,
                                       fill=fill,
                                       alpha=0.5))



plt.xlim(-0.5, len(corr.columns) - 0.5)
plt.ylim(-0.5, len(corr.columns) - 0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xticks(np.arange(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(np.arange(len(corr.columns)), corr.columns)
plt.title('Correlogram')
plt.show()



%md Fit the regression model

from math import sqrt

import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# turn on autologging
mlflow.sklearn.autolog(log_input_examples=True)

# apply the Standard Scaler to all our input columns
std_ct = ColumnTransformer(transformers=[("scaler", StandardScaler(), ["total_bedrooms", "total_rooms", "housing_median_age", "latitude", "longitude", "median_income", "population", "ocean_proximity", "households"])])

# pipeline to transform inputs and then pass results to the linear regression model
lr_pl = Pipeline(steps=[
  ("tx_inputs", std_ct),
  ("lr", LinearRegression() )
])

# fit our model
lr_mdl = lr_pl.fit(X_train, y_train)

# evaluate the test set
predicted = lr_mdl.predict(X_test)
test_r2 = r2_score(y_test, predicted)
test_mse = mean_squared_error(y_test, predicted)
test_rmse = sqrt(test_mse)
test_mape = mean_absolute_percentage_error(y_test, predicted)
print("Test evaluation summary:")
print(f"R^2: {test_r2}")
print(f"MSE: {test_mse}")
print(f"RMSE: {test_rmse}")
print(f"MAPE: {test_mape}")


# Check the model result

lr_mdl


import pandas as pd
import numpy as np
from scipy import stats

# Extracting coefficients and intercept
coefficients = np.append([lr_mdl.named_steps['lr'].intercept_], lr_mdl.named_steps['lr'].coef_)
coefficient_names = ['Intercept'] + X_train.columns.to_list()

# Calculating standard errors and other statistics (this is a simplified example)
# In a real scenario, you might need to calculate these values more rigorously
n_rows, n_cols = X_train.shape
X_with_intercept = np.append(np.ones((n_rows, 1)), X_train, axis=1)
var_b = test_mse * np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept)).diagonal()
standard_errors = np.sqrt(var_b)
t_values = coefficients / standard_errors
p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(X_with_intercept) - 1))) for i in t_values]

# Creating a DataFrame for display
summary_df = pd.DataFrame({'Coefficient': coefficients,
                           'Standard Error': standard_errors,
                           't-value': t_values,
                           'p-value': p_values},
                          index=coefficient_names)

# Print the DataFrame
print(summary_df)



%md Feature Importance

import matplotlib.pyplot as plt

# Plotting the feature importances
plt.figure(figsize=(10, 6))
y_pos = np.arange(len(coefficient_names))
plt.bar(y_pos, coefficients, align='center', alpha=0.7)
plt.xticks(y_pos, coefficient_names, rotation=45)
plt.ylabel('Coefficient Size')
plt.title('Coefficients in Linear Regression')

plt.show()



