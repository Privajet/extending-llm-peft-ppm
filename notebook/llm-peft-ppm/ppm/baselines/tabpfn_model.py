from sklearn.metrics import accuracy_score, mean_squared_error
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.many_class import ManyClassClassifier


def run_tabpfn_baseline(train_log, test_log, random_state: int):
    """
    - Klassifikation: next_activity
    - Regression: remaining_time, time_to_next_event
    """

    max_n_train = 50000
    max_n_test = 2000
    
    df_train = train_log.dataframe
    df_test = test_log.dataframe
    
    if len(df_train) > max_n_train:
        df_train = df_train.sample(n=max_n_train, random_state=random_state).reset_index(drop=True)
        
    if len(df_test) > max_n_test:
        df_test = df_test.sample(n=max_n_test, random_state=random_state).reset_index(drop=True)

    feature_cols = list(train_log.features.categorical) + list(train_log.features.numerical)

    X_train = df_train[feature_cols].to_numpy()
    X_test = df_test[feature_cols].to_numpy()

    metrics = {}

    # Classification: next_activity
    y_train_cls = df_train["next_activity"].to_numpy()
    y_test_cls = df_test["next_activity"].to_numpy()

    base_clf = TabPFNClassifier()
    clf = ManyClassClassifier(
        estimator=base_clf,
        alphabet_size=10,         
        n_estimators=None,           
        n_estimators_redundancy=4,   
        random_state=random_state,
        verbose=0,
        codebook_config=None,
        row_weighting_config=None,
        aggregation_config=None,
    )
    clf.fit(X_train, y_train_cls)
    y_pred_cls = clf.predict(X_test)

    metrics["test_next_activity_acc"] = float(accuracy_score(y_test_cls, y_pred_cls))

    # Regression: remaining_time 
    y_train_rt = df_train["remaining_time"].astype(float).to_numpy()
    y_test_rt = df_test["remaining_time"].astype(float).to_numpy()

    reg_rt = TabPFNRegressor()
    reg_rt.fit(X_train, y_train_rt)
    pred_rt = reg_rt.predict(X_test)

    metrics["test_remaining_time_mse"] = float(mean_squared_error(y_test_rt, pred_rt))

    # Regression: time_to_next_event 
    y_train_nt = df_train["time_to_next_event"].astype(float).to_numpy()
    y_test_nt = df_test["time_to_next_event"].astype(float).to_numpy()

    reg_nt = TabPFNRegressor()
    reg_nt.fit(X_train, y_train_nt)
    pred_nt = reg_nt.predict(X_test)

    metrics["test_time_to_next_event_mse"] = float(mean_squared_error(y_test_nt, pred_nt))

    return metrics