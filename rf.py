from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def rf(X_train, X_test, mode_train, mode_test, campaigns, test_campaigns, n_estimators=100):
    rf = RandomForestClassifier(n_estimators=n_estimators)
    smote = SMOTE()
    rus = RandomUnderSampler()
    X_train_balanced, mode_train_balanced = smote.fit_resample(X_train, mode_train)
    # X_train_balanced, mode_train_balanced = rus.fit_resample(X_train_balanced, mode_train_balanced)

    rf.fit(X_train_balanced, mode_train_balanced)

    if len(test_campaigns) == 1:
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(mode_test['Mode'], y_pred)
        precision = precision_score(mode_test['Mode'], y_pred)
        recall = recall_score(mode_test['Mode'], y_pred)
    else:
        X_test_test = X_test[X_test['campaign'] == test_campaigns[0]]
        X_test_mode = mode_test[mode_test['campaign'] == test_campaigns[0]]
        
        y_pred = rf.predict(X_test_test)
        accuracy = accuracy_score(X_test_mode['Mode'], y_pred)
        precision = precision_score(X_test_mode['Mode'], y_pred)
        recall = recall_score(X_test_mode['Mode'], y_pred)

    if len(campaigns) == 1:
        train_pred = rf.predict(X_train)
        train_accuracy = accuracy_score(mode_train['Mode'], train_pred)
        train_precision = precision_score(mode_train['Mode'], train_pred)
        train_recall = recall_score(mode_train['Mode'], train_pred)
    else:
        X_train_test = X_train[X_train['campaign'] == campaigns[0]]
        X_train_mode = mode_train[mode_train['campaign'] == campaigns[0]]

        train_pred = rf.predict(X_train_test)
        train_accuracy = accuracy_score(X_train_mode['Mode'], train_pred)
        train_precision = precision_score(X_train_mode['Mode'], train_pred)
        train_recall = recall_score(X_train_mode['Mode'], train_pred)

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