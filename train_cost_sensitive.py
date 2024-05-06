import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

# Assuming df is your loaded dataset
# df = pd.read_csv('your_dataset.csv')``

def preprocess_data(df, dropna=True, drop_columns=[]):
    if dropna:
        df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    df = df.drop(columns=drop_columns)
    return df

def create_pipeline(estimator, use_scaler=True):
    steps = []
    if use_scaler:
        steps.append(('scaler', StandardScaler()))
    steps.append(('estimator', estimator))
    pipeline = Pipeline(steps)
    return pipeline

def perform_CV(X_train, y_train, estimator, cv_strategy, detailed_results_file):
    pipeline = create_pipeline(estimator)
    cross_validate_results = cross_validate(pipeline,X_train, y_train, cv=cv_strategy, scoring=['f1','roc_auc','precision','recall'], )
    
    
    results = pd.DataFrame(cross_validate_results)
    
    # # Save the summary results to a CSV file
    # summary_results = results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    # summary_results.to_csv(summary_results_file, index=False)
    
    # Extract only the detailed results for the best parameter set
    # columns_to_keep = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    # columns_to_keep.extend([key for key in grid_search.cv_results_ if key.startswith('split')])
    # results_filtered = results[columns_to_keep]
    
    results.to_csv(detailed_results_file, index=False)
    
    # return cross_validate_results.best_score_


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions)
    return f1

if __name__ == "__main__":
    class_weights = [
    # {0: 1, 1: 1},    # equal weight
    {0: 1, 1: 2},    # increasing weight for class 1
    {0: 1, 1: 5},
    {0: 1, 1: 10},
    {0: 1, 1: 20},
    {0: 1, 1: 50},
    {0: 1, 1: 100}
    ]

 

    df = pd.read_csv('data_preprocessed.csv')
    df_processed = preprocess_data(df, drop_columns=['relationship_Wife'])

    # Split dataset into training and test data
    X = df_processed.drop('income_>50K', axis=1)
    y = df_processed['income_>50K']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Define the cross-validation strategy
    cv_strategy = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # Search for the best model and parameters for each estimator
    best_models = {}
    best_parameters = {}
    for weights in class_weights:
        print(weights)
        rfc = RandomForestClassifier(class_weight=weights, criterion='entropy', max_depth=None, max_features='sqrt', min_samples_leaf=2, min_samples_split=10, n_estimators=300)
        detailed_results_file = f'RFC_CSL_{str(weights[1])}.csv'
        perform_CV(X_train, y_train, rfc, cv_strategy, detailed_results_file)
