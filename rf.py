from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

def rf(X, mode):
    X_train, X_test, y_train, y_test = train_test_split(X, mode, test_size=0.3)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    results = {'Accuracy': accuracy,
               'Precision': precision,
               'Recall': recall,
               'y_true' : y_test,
               'y_pred' : y_pred,
               'model': rf}
    return results