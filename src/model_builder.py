from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def train_logistic_regression(train, test, features):
    scaler = StandardScaler()
    X_train_lr = scaler.fit_transform(train[features])
    X_test_lr = scaler.transform(test[features])
    y_train = train["target"]

    lr_model = LogisticRegression(C=0.5, penalty="l2", solver="lbfgs", max_iter=1000)
    lr_model.fit(X_train_lr, y_train)

    train_out = train.copy()
    test_out = test.copy()
    
    train_out["prob_lr"] = lr_model.predict_proba(X_train_lr)[:, 1]
    test_out["prob_lr"] = lr_model.predict_proba(X_test_lr)[:, 1]
    
    return train_out, test_out

def train_xgboost(train, test, features):
    y_train = train["target"]
    y_test = test["target"]

    # Using the best parameters found from your RandomizedSearchCV
    best_params = {
        'subsample': 0.8, 'reg_lambda': 1.5, 'reg_alpha': 0.1, 
        'n_estimators': 200, 'min_child_weight': 3, 'max_depth': 4, 
        'learning_rate': 0.05, 'gamma': 0.1, 'colsample_bytree': 0.8
    }

    xgb_model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42
    )

    xgb_model.fit(
        train[features], y_train,
        eval_set=[(test[features], y_test)],
        verbose=False
    )

    train_out = train.copy()
    test_out = test.copy()
    
    train_out["prob_xgb"] = xgb_model.predict_proba(train[features])[:, 1]
    test_out["prob_xgb"] = xgb_model.predict_proba(test[features])[:, 1]
    
    return train_out, test_out
