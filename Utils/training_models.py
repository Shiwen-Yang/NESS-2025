import os
import json
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
def train_xgb(X_train, y_train, X_val, y_val, params, seed = None):
    
    xgb_params_untrainable = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist',
        'verbosity': 0,
        'seed': seed
    }
        
    params.update(xgb_params_untrainable)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=False  # silent logs
    )

    # Predict + threshold tuning
    probs = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    thresholds = np.linspace(0.1, 0.9, 200)
    f1s = [f1_score(y_val, probs > t) for t in thresholds]

    best_f1 = max(f1s)
    best_threshold = thresholds[np.argmax(f1s)]
    best_iterations = model.best_iteration    
    stats = {'f1': best_f1, 'threshold': best_threshold, 'iteration': best_iterations}
    
    return(probs, model, stats)

def sample_xgb_hyperparams(trial):
    return {
        'max_depth': trial.suggest_int('max_depth', 1, 15),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.3, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
        'eta': trial.suggest_float('learning_rate', 0.005, 0.1),
        'lambda': trial.suggest_float('lambda', 0.01, 25.0, log=True),
        'alpha': trial.suggest_float('alpha', 0.01, 20.0, log=True),
    }
    
def predict_xgb(model, test_df):
    dtest = xgb.DMatrix(test_df)
    return model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
    
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
def train_lgb(X_train, y_train, X_val, y_val, params, seed=None):
    # Set or override fixed parameters
    lgb_params_untrainable = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting': 'gbdt',
        'device': 'gpu',
        'verbose': -1,
        'seed': seed if seed is not None else 42
    }
    
    params.update(lgb_params_untrainable)

    # Prepare datasets
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # Train model
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False)
        ]
    )

    # Predict + threshold tuning
    probs = model.predict(X_val, num_iteration=model.best_iteration)
    thresholds = np.linspace(0.1, 0.9, 200)
    f1s = [f1_score(y_val, probs > t) for t in thresholds]

    best_f1 = max(f1s)
    best_threshold = thresholds[np.argmax(f1s)]
    best_iterations = model.best_iteration

    stats = {
        'f1': best_f1,
        'threshold': best_threshold,
        'iteration': best_iterations
    }

    return probs, model, stats

def sample_lgb_hyperparams(trial):
    return {
        'num_leaves': trial.suggest_int('num_leaves', 50, 90),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 20.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 25.0, log=True),
    }
    
def predict_lgb(model, test_df):
    return model.predict(test_df, num_iteration=model.best_iteration)

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
def train_cat(X_train, y_train, X_val, y_val, params, seed=None):
    # Set or override fixed parameters
    cat_params_untrainable = {
        'iterations': 2000,
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'verbose': False,
        'random_seed': seed if seed is not None else 42,
        # 'task_type': 'GPU'
    }

    params.update(cat_params_untrainable)

    # Create CatBoost pools
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    # Initialize and train model
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

    # Predict + threshold tuning
    probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 200)
    f1s = [f1_score(y_val, probs > t) for t in thresholds]

    best_f1 = max(f1s)
    best_threshold = thresholds[np.argmax(f1s)]
    best_iterations = model.get_best_iteration()

    stats = {
        'f1': best_f1,
        'threshold': best_threshold,
        'iteration': best_iterations
    }

    return probs, model, stats

def sample_cat_hyperparams(trial):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 25.0, log=True),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
        # 'bootstrap_type': 'Bayesian',  # Optional: for advanced users
        # 'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),  # Only if bootstrap_type is Bayesian
    }

def predict_cat(model, test_df):
    return model.predict_proba(test_df)[:, 1]

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
def train_hgb(X_train, y_train, X_val, y_val, params, seed=None):
    # Set or override fixed parameters
    hgbc_params_untrainable = {
        'early_stopping': True,
        'validation_fraction': 0.1,
        'random_state': seed if seed is not None else 42,
        'verbose': 0
    }
    params.update(hgbc_params_untrainable)

    # Instantiate and train the model
    model = HistGradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    # Predict + threshold tuning
    probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 200)
    f1s = [f1_score(y_val, probs > t) for t in thresholds]

    best_f1 = max(f1s)
    best_threshold = thresholds[np.argmax(f1s)]
    best_iterations = getattr(model, 'n_iter_', None)  # Not as meaningful as boosting libs

    stats = {
        'f1': best_f1,
        'threshold': best_threshold,
        'iteration': best_iterations
    }

    return probs, model, stats

def sample_hgb_hyperparams(trial):
    return {
        'max_iter': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 90),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-3, 25.0, log=True)
    }
    
def predict_hgb(model, test_df):
    return model.predict_proba(test_df)[:, 1]
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
def run_cv_evaluation_single_model(X, y, params, train_model_fn, kfoldcv=5, drop=[], test_df = None, predict_fn = None, seed = None):

    skf = StratifiedKFold(n_splits=kfoldcv, shuffle=True, random_state=seed)

    results = []
    probs_test_list = []

    for train_idx, val_idx in skf.split(X, y):
        X_train = X.iloc[train_idx].drop(columns=drop, errors='ignore')
        X_val = X.iloc[val_idx].drop(columns=drop, errors='ignore')
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        _, model, stats = train_model_fn(X_train, y_train, X_val, y_val, params)
        results.append(stats)
        
        if test_df is not None:
            probs_test = predict_fn(model, test_df)
            probs_test_list.append(probs_test)

    if test_df is not None:
        avg_probs_test = pd.DataFrame(probs_test_list).T.mean(axis=1)
        return pd.DataFrame(results), avg_probs_test
    else:
        return pd.DataFrame(results)

def objective_single_model(trial, full_train_df, target, train_model_fn, params_trial_fn, kfoldcv=5, drop=[], seed = None):
    params = params_trial_fn(trial)
    cv_df = run_cv_evaluation_single_model(full_train_df, target, params, train_model_fn, kfoldcv=kfoldcv, drop=drop, seed = seed)
    trial.set_user_attr('cv_results', cv_df)
    return cv_df['f1'].mean()


#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
def save_settings(train_df, test_df, test_id, test_pred, best_params, threshold_for_f1, output_dir, model_name):
    
    final_preds = (test_pred > threshold_for_f1).astype(int)

    submission = pd.DataFrame({
        'claim_number': test_id,
        'fraud': final_preds
    })

    train_csv_path = os.path.join(output_dir, 'train_2025.csv')
    test_csv_path = os.path.join(output_dir, 'test_2025.csv')
    param_json_path = os.path.join(output_dir, f'param_{model_name}_temp.json')
    submission_csv_path = os.path.join(output_dir, f'submission_{model_name}.csv')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save DataFrames
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    submission.to_csv(submission_csv_path, index=False)
    

    # Save best_params as JSON
    best_params.update({'mean_threshold': threshold_for_f1})
    
    with open(param_json_path, 'w') as f:
        json.dump(best_params, f, indent=4)


#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
def read_settings(path_to_settings):
    with open(path_to_settings, 'r') as f:
        settings = json.load(f)

    threshold = settings.pop('mean_threshold', None) 
    return settings, threshold 
