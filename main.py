from src.predictor import SimpleNNPredictor
from data.read_data import read_data

from sklearn.model_selection import train_test_split
import numpy as np


generateValData = False
hp_opt = False
# Read data
X, y = read_data()

if generateValData:
    # Artificially split data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=123
    )
else:
    X_train, X_val, y_train, y_val = X, X, y, y

# Instantiate predictor
model = SimpleNNPredictor(hp_opt=hp_opt)

# Fit model with unscaled data
model.train(X_train, y_train)

# Predict outputs for train and validation set
y_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# Show accuracy metrics
train_acc = np.average((y_pred - y_train) ** 2)
val_acc = np.average((y_val_pred - y_val) ** 2)
print("Train MSE:" + str(train_acc))
print("Validation MSE:" + str(val_acc))

# Make predictions
model.run_validate()
model.run_test()