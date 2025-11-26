from sklearn.metrics import accuracy_score, mean_squared_error
from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor


def run_sap_rpt_baseline(train_log, test_log, random_state: int):
    """
    - Classification: next_activity
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

    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]

    metrics = {}

    # Classification: next_activity
    y_train_cls = df_train["next_activity"]
    y_test_cls = df_test["next_activity"]

    clf = SAP_RPT_OSS_Classifier(
        max_context_size=1024, 
        bagging=1,
    )
    clf.fit(X_train, y_train_cls)
    y_pred_cls = clf.predict(X_test)

    metrics["test_next_activity_acc"] = float(accuracy_score(y_test_cls, y_pred_cls))

    # Regression: remaining_time 
    y_train_rt = df_train["remaining_time"].astype(float)
    y_test_rt = df_test["remaining_time"].astype(float)

    reg_rt = SAP_RPT_OSS_Regressor(
        max_context_size=1024,
        bagging=1,
    )
    reg_rt.fit(X_train, y_train_rt)
    pred_rt = reg_rt.predict(X_test)

    metrics["test_next_remaining_time_loss"] = float(mean_squared_error(y_test_rt, pred_rt))

    # Regression: time_to_next_event 
    y_train_nt = df_train["time_to_next_event"].astype(float)
    y_test_nt = df_test["time_to_next_event"].astype(float)

    reg_nt = SAP_RPT_OSS_Regressor(
        max_context_size=1024,
        bagging=1,
    )
    reg_nt.fit(X_train, y_train_nt)
    pred_nt = reg_nt.predict(X_test)

    metrics["test_next_time_to_next_event_loss"] = float(mean_squared_error(y_test_nt, pred_nt))

    metrics["y_true_pred_dump"] = {
        "y_true_next_activity": y_test_cls.to_numpy(),
        "y_pred_next_activity": y_pred_cls,
        "y_true_remaining_time": y_test_rt.to_numpy(),
        "y_pred_remaining_time": pred_rt,
        "y_true_time_to_next_event": y_test_nt.to_numpy(),
        "y_pred_time_to_next_event": pred_nt,
    }
        
    return metrics