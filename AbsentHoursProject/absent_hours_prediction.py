import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

# Load your dataset
df = pd.read_csv("data/MFGEmployees.csv")   

#1)Preprocessing
# Drop unnecessary columns
df = df.drop(columns=['EmployeeNumber', 'Surname', 'GivenName'])
df = df.dropna().drop_duplicates()
df = df[(df['Age'] > 18) & (df['Age'] < 65)]

# Prepare features and target
X = df.drop(columns='AbsentHours')
y = df['AbsentHours']

# Identify columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

#2)Model Creation
# Pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42, tol=0.0001, subsample=0.8, n_estimators=100, min_samples_split=2, min_samples_leaf=4, max_features=None, max_depth=3, loss='huber', learning_rate=0.1, criterion='squared_error', alpha=0.9))
])

#4)Train/test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)


#5)Save model
with open("absent_hours_prediction.pkl", "wb") as f:
    pickle.dump(pipeline, f)

