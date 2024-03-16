import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("dataset.csv")

# Handling null values
imputer = SimpleImputer(fill_value=0)
df_imputed = df.copy()
df_imputed[['Age', 'Annual_Income', 'Credit_Score', 'Loan_Amount', 'Number_of_Open_Accounts']] = imputer.fit_transform(df[['Age', 'Annual_Income', 'Credit_Score', 'Loan_Amount', 'Number_of_Open_Accounts']])

# Splitting the data into features and target
X = df_imputed.drop('Loan_Approval', axis=1)
y = df_imputed['Loan_Approval']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

# Training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print(y_pred)
