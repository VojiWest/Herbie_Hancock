from sklearn.linear_model import LogisticRegression
import numpy as np


def make_logistic_regression(X, y):
    logistic_regressor = LogisticRegression(max_iter = 50000)
    logistic_regressor.fit(X, y)

    return logistic_regressor

def run_logistic_regression(logistic_regressor, last_X, extension_length = 400):
    top_preds = []
    all_preds = []
    curr_input = last_X

    for i in range(extension_length):
        reshaped_input = curr_input.reshape(1, -1)
        # Predict the next timestep
        pred = logistic_regressor.predict(reshaped_input)
        top_preds.append(pred)
        
        all_predictiosn = logistic_regressor.predict_proba(reshaped_input)
        all_preds.append(all_predictiosn)

        # Update the input
        curr_input = np.append(curr_input[1:], pred)

    return np.array(top_preds), np.array(all_preds)
