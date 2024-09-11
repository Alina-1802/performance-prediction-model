import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np

pd.set_option('display.max_columns', None)

# 1. Data import
df = pd.read_csv('garments_worker_productivity.csv', delimiter=',', header=0)
df['date'] = pd.to_datetime(df['date'])

# 2. Basic statistics
describe_stats = (df.iloc[:, 5:]).describe()
describe_stats.to_excel('statistics.xlsx', index=True)

# plots
plt.rcParams.update({'font.size': 26})
columns = df.columns
for column in columns:
    plt.figure(figsize=(10, 8))
    if column == 'date':
        plt.hist(df[column], bins=30, edgecolor='k')
        plt.xlabel('Data')
        plt.ylabel('Częstość')
        plt.title(f"Rozkład zmiennej {column}")
        plt.xticks(rotation=45)
        plt.tight_layout()
    elif df[column].dtype == 'object':
        sns.countplot(x=column, data=df, edgecolor='k')
        plt.xlabel('Kategorie')
        plt.ylabel('Częstość')
        plt.title(f"Rozkład zmiennej {column}")
        plt.xticks(rotation=45)
    else:
        plt.hist(df[column], bins=30, edgecolor='k')
        plt.xlabel('Wartości zmiennej')
        plt.ylabel('Częstość')
        plt.title(f"Rozkład zmiennej {column}")
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(f"rozkład_{column}.png")
    plt.close()

# 3. Convert non numeric columns
def convert_non_numeric(data):
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data['date'].dt.month
    data['day_of_month'] = data['date'].dt.day
    data = data.drop('date', axis=1)
    categorical_columns = ['quarter', 'department', 'day']
    for column in categorical_columns:
        data[column] = data[column].str.strip()
    data = pd.get_dummies(data, columns=categorical_columns)
    return data

df_converted = convert_non_numeric(df)
df_converted.to_excel('converted.xlsx', index=True)

# 4. Handle NA records
#df_complete = df_converted.fillna(0)
#df_complete = df_converted.dropna()
def knn_impute(data):
    imputer = KNNImputer(n_neighbors=5)
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return data_imputed

df_complete = knn_impute(df_converted)
df_complete.to_excel('withoutNA.xlsx', index=True)

# 5. Normalize (z-score)

df_norm = (df_complete - df_complete.mean()) / df_complete.std()
df_norm.to_excel('normalized.xlsx', index=True)
#df_norm = df_complete


# 6. Remove outliers
def remove_outliers(data, columns):
    for column in columns:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data

from scipy import stats
def remove_outliers_zscore(data, columns, threshold=3):
    z_scores = np.abs(stats.zscore(data[columns]))
    data = data[(z_scores < threshold).all(axis=1)]
    return data

df_without_outliers = remove_outliers_zscore(df_norm, df_norm.columns)
df_without_outliers.to_excel('without_outliers.xlsx', index=True)
#df_without_outliers = df_norm

# 7. Correlations

correlation_matrix = df_without_outliers.corr()
correlation_matrix.to_excel('correlations.xlsx', index=True)

# 8. Shuffle records and split the data into training and test data

x1 = df_without_outliers.drop(columns=['actual_productivity'])
x2 = df_without_outliers[['team', 'targeted_productivity', 'smv', 'idle_men', 'no_of_style_change' ]]
x3 = df_without_outliers.drop(columns=['actual_productivity','day_Saturday', 'quarter_Quarter2', 'day_Tuesday', 'day_Monday', 'day_Wednesday', 'day_Sunday', 'day_Thursday', 'department_finishing'])
y = df_without_outliers.actual_productivity

x_train_indices, x_test_indices, y_train, y_test = train_test_split(df_without_outliers.index, y, test_size=0.3, random_state=1)

x1_train = x1.loc[x_train_indices]
x1_test = x1.loc[x_test_indices]

x2_train = x2.loc[x_train_indices]
x2_test = x2.loc[x_test_indices]

x3_train = x3.loc[x_train_indices]
x3_test = x3.loc[x_test_indices]

# 9. Training
model1 = LinearRegression()
model1.fit(x1_train, y_train)

model2 = LinearRegression()
model2.fit(x2_train, y_train)

model3 = LinearRegression()
model3.fit(x3_train, y_train)

intercept1 = model1.intercept_
coefficients1 = model1.coef_
equation1 = f'y = {coefficients1[0]} * X + {intercept1}'
print("Równanie liniowe modelu 1: ")
print(equation1)

intercept2 = model2.intercept_
coefficients2 = model2.coef_
equation2 = f'y = {coefficients2[0]} * X + {intercept2}'
print("Równanie liniowe modelu 2: ")
print(equation2)

intercept3 = model3.intercept_
coefficients3 = model3.coef_
equation3 = f'y = {coefficients3[0]} * X + {intercept3}'
print("Równanie liniowe modelu 3: ")
print(equation3)

# 10. Models evaluation

y1_pred = model1.predict(x1_test)
y2_pred = model2.predict(x2_test)
y3_pred = model3.predict(x3_test)

# Plot actual vs predicted values

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y1_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')  # Diagonal line
plt.title('Actual vs Predicted Values 1')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
#plt.show()
plt.savefig("act_pre_model1")

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y2_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')  # Diagonal line
plt.title('Actual vs Predicted Values 2')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
#plt.show()
plt.savefig("act_pre_model2")

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y3_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')  # Diagonal line
plt.title('Actual vs Predicted Values 3')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
#plt.show()
plt.savefig("act_pre_model3")


# Calculate residuals

residuals1 = y_test - y1_pred
residuals2 = y_test - y2_pred
residuals3 = y_test - y3_pred

#Plot histogram of residuals
plt.figure(figsize=(10, 8))
plt.hist(residuals1, bins=30, edgecolor='k')
plt.title('Histogram of Residuals 1')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
#plt.show()
plt.savefig("res_model1")

plt.figure(figsize=(10, 8))
plt.hist(residuals2, bins=30, edgecolor='k')
plt.title('Histogram of Residuals 2')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
#plt.show()
plt.savefig("res_model2")

plt.figure(figsize=(10, 8))
plt.hist(residuals3, bins=30, edgecolor='k')
plt.title('Histogram of Residuals 3')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
#plt.show()
plt.savefig("res_model3")

from sklearn.metrics import r2_score, mean_absolute_error

# Calculate mean squared error
mse1 = mean_squared_error(y_test, y1_pred)
mse2 = mean_squared_error(y_test, y2_pred)
mse3 = mean_squared_error(y_test, y3_pred)

print("Mean Squared Error 1:", mse1)
print("Mean Squared Error 2:", mse2)
print("Mean Squared Error 3:", mse3)

# Calculate R^2 Score
r2_1 = r2_score(y_test, y1_pred)
r2_2 = r2_score(y_test, y2_pred)
r2_3 = r2_score(y_test, y3_pred)

print("R^2 Score 1:", r2_1)
print("R^2 Score 2:", r2_2)
print("R^2 Score 3:", r2_3)

# Calculate Mean Absolute Error (MAE)
mae1 = mean_absolute_error(y_test, y1_pred)
mae2 = mean_absolute_error(y_test, y2_pred)
mae3 = mean_absolute_error(y_test, y3_pred)

print("Mean Absolute Error 1:", mae1)
print("Mean Absolute Error 2:", mae2)
print("Mean Absolute Error 3:", mae3)

# Calculate Root Mean Squared Error (RMSE)
rmse1 = np.sqrt(mse1)
rmse2 = np.sqrt(mse2)
rmse3 = np.sqrt(mse3)

print("Root Mean Squared Error 1:", rmse1)
print("Root Mean Squared Error 2:", rmse2)
print("Root Mean Squared Error 3:", rmse3)