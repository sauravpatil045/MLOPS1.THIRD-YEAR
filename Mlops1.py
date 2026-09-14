import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10],
    "Attendance": [60,85,70,95,65,80,90,75,88,72],
    "Marks": [32,42,47,58,63,67,76,79,88,91],
}
# Convert dict into Dataframe
df = pd.DataFrame(data)
print(df)
print('First Five Rows:')
df.head()
print('Shape of Dataset:')
df.shape
print('Information:')
df.info()
print('Stastical Analysis:\n', df.describe())
print('Check for null values:')
df.isnull().sum()
# Duplicate Values
df.duplicated().sum()
# Correction Heatmap
import seaborn as sns

df_corr = df.corr()

plt.figure(figsize= ( 6, 4))
sns.heatmap(df_corr, annot=True, cmap='Blues')

plt.title('Correction Heatmap')
# Feature Selection
x=df[['Study_Hours']] # Independent Variable
y=df['Marks'] # Dependent Variable
x
# Split dataset into training and testing set 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_test
# Import Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Train the model
model.fit(x_train, y_train)
# Predict the values
y_pred = model.predict(x_test)
y_pred
# Print Equation
print('Intercept:', model.intercept_)
print('Slope:', model.coef_[0])
# SLR EQUATION
Marks_predicted_from_equation = (6.48 * df['Study_Hours']) + 28.27
# Evaluation
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print('RMSE:',round(rmse,3))
print('R2 Score:', round(r2, 2))
# Regression line
plt.figure(figsize=(7, 5))
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x_test, y_pred, color='red', label='Regression Line')
plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
residuals = y_test - y_pred #Calculate Residual 

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
x =df["Study_Hours"].values #Independent Variable (features)
y =df["Marks"].values #Dependent Variables (Target)
x
# Calculate Mean
x_mean = np.mean(x)
y_mean = np.mean(y)
x_mean
# Calculate Slope
m = np.sum((x-x_mean)*(y-y_mean))/np.sum((x-x_mean)**2)
# Calculate intercept
b =y_mean - (m*x_mean)
print("Slope =", m)
print("Intersept =", b)
# Prediction
y_pred_numpy =m*x + b
y_pred_numpy
# Feature Selection
x=df[['Study_Hours', 'Attendance']] # Independent Variable
y=df['Marks'] # Dependent Variable
# Spilit Database
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_train
model=LinearRegression()
model.fit(x_train, y_train)
# Prediction
y_pred =model.predict(x_test)
y_pred
# Model Parameters
print('Intercept:', model.intercept_)
print("Coefficients",)
coeff=pd.DataFrame({
    'Feature': x.columns, 'Coefficient': model.coef_ })

print(coeff)     
# Evaluation Metrics
from sklearn.metrics import mean_absolute_error

print('Metrics Values Of MLR:')
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print('MAE:', mae)
print('RMSE:', round(rmse, 2))
print('R2 Score:', round(r2, 2))
# Actual vs Predicted Plot
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, color='purple')

# Perfect Prediction Line
plt.plot([y.min(), y.max()],
         [y_pred.min(), y_pred.max()], color='red')

plt.xlabel('Actual Marks')
plt.ylabel('Predicted Marks')
plt.title('Actual vs Predicted (MLR)')
plt.show()
residuals = y_test - y_pred #Calculate Residual

plt.scatter(y_pred, residuals, color='Orange')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot(MLR)')
plt.show()