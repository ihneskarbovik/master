from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

def rf(X_train, X_test, mode_train, mode_test):
    rf = RandomForestClassifier()
    rf.fit(X_train, mode_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(mode_test, y_pred)
    precision = precision_score(mode_test, y_pred)
    recall = recall_score(mode_test, y_pred)
    results = {'Accuracy': accuracy,
               'Precision': precision,
               'Recall': recall,
               'y_true' : mode_test,
               'y_pred' : y_pred,
               'model': rf}
    return results