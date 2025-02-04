from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

def rf(X_train, X_test, mode_train, mode_test, n_estimators=100):
    rf = RandomForestClassifier(n_estimators=n_estimators)
    rf.fit(X_train, mode_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(mode_test, y_pred)
    precision = precision_score(mode_test, y_pred)
    recall = recall_score(mode_test, y_pred)

    train_pred = rf.predict(X_train)
    train_accuracy = accuracy_score(mode_train, train_pred)
    train_precision = precision_score(mode_train, train_pred)
    train_recall = recall_score(mode_train, train_pred)

    results = {'Accuracy': round(accuracy, 3),
               'Precision': round(precision, 3),
               'Recall': round(recall, 3),
               'y_true' : mode_test,
               'y_pred' : y_pred,
               'train_true': mode_train,
               'train_pred': train_pred,
               'Accuracy_train': round(train_accuracy, 3),
               'Precision_train': round(train_precision, 3),
               'Recall_train': round(train_recall, 3),
               'model': rf}
    return results