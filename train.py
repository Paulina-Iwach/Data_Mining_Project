import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

# Assuming df is your loaded dataset
# df = pd.read_csv('your_dataset.csv')

def preprocess_data(df, dropna=True, drop_columns=[]):
    if dropna:
        df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    df = df.drop(columns=drop_columns)
    return df

def create_pipeline(estimator, use_scaler=True):
    steps = []
    if use_scaler:
        steps.append(('scaler', MinMaxScaler()))
    steps.append(('estimator', estimator))
    pipeline = Pipeline(steps)
    return pipeline

def perform_grid_search(X_train, y_train, estimator, param_grid, cv_strategy, detailed_results_file):
    pipeline = create_pipeline(estimator)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv_strategy, scoring='f1', n_jobs=-1, return_train_score=False)
    # grid_search = RandomizedSearchCV(pipeline, param_grid, cv=cv_strategy, scoring='f1', n_jobs=-1, return_train_score=False)
    grid_search.fit(X_train, y_train)
    
    results = pd.DataFrame(grid_search.cv_results_)
    
    # # Save the summary results to a CSV file
    # summary_results = results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    # summary_results.to_csv(summary_results_file, index=False)
    
    # Extract only the detailed results for the best parameter set
    columns_to_keep = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    columns_to_keep.extend([key for key in grid_search.cv_results_ if key.startswith('split')])
    results_filtered = results[columns_to_keep]
    
    results_filtered.to_csv(detailed_results_file, index=False)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions)
    return f1

if __name__ == "__main__":
    param_grid_rfc = {
        'estimator__n_estimators': [100, 200, 300, 500, 1000],  # Number of trees in the forest
        'estimator__max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at every split
        'estimator__max_depth': [10, 20, 30, 40, 50, None],  # Maximum number of levels in tree
        'estimator__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'estimator__min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
        'estimator__criterion': ['gini', 'entropy', 'log_loss'] # Method of selecting samples for training each tree
    }

    param_grid_lda = {
        # 'estimator__solver': ['svd', 'lsqr', 'eigen'],
        # 'estimator__n_components': [1, 2, 3, ...], # Uncomment if dimensionality reduction is needed
        # Shrinkage can only be used with the 'lsqr' and 'eigen' solvers
        'estimator__shrinkage': [None, 'auto', 0.99, 0.8, 0.6, 0.4, 0.2, 0.01]  # or a list np.linspace(0, 1, num=10)
    }

    param_grid_qda = {
        'estimator__reg_param': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Regularization parameter
    }

    param_grid_lr = [
        {'estimator__C': [0.01, 0.1, 1, 10, 100], 'estimator__solver': ['newton-cg', 'lbfgs', 'sag'], 'estimator__penalty': ['l2', 'none'], 'estimator__max_iter': [100, 200, 300]},
        {'estimator__C': [0.01, 0.1, 1, 10, 100], 'estimator__solver': ['liblinear'], 'estimator__penalty': ['l1', 'l2'], 'estimator__max_iter': [100, 200, 300]},
        {'estimator__C': [0.01, 0.1, 1, 10, 100], 'estimator__solver': ['saga'], 'estimator__penalty': ['l1', 'l2', 'elasticnet', 'none'], 'estimator__max_iter': [100, 200, 300]}
    ]

    param_grid_dtr = {
        'estimator__max_depth': [None, 10, 20, 30, 40, 50],
        'estimator__min_samples_split': [2, 5, 10, 20],
        'estimator__min_samples_leaf': [1, 2, 4, 10],
        'estimator__max_features': ['auto', 'sqrt', 'log2', None],
        'estimator__criterion': ['gini', 'entropy']
    }

    param_grid_knn = {
        'estimator__n_neighbors': [3, 5, 7, 10, 15],
        'estimator__weights': ['uniform', 'distance'],
        # 'estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'estimator__p': [1, 2, 3]  # Typically, 1 (Manhattan) or 2 (Euclidean) are used, but you can explore others
    }


    # Define your estimators and their parameter grids
    estimators_and_parameters = {
        'LDA': (LDA(), param_grid_lda),
        'QDA': (QDA(), param_grid_qda),
        'LR': (LogisticRegression(), param_grid_lr),
        'DTR': (DecisionTreeClassifier(), param_grid_dtr),
        # 'KNN': (KNeighborsClassifier(), param_grid_knn),
        # 'SVM': (SVC(), param_grid_svm),
        'RFC': (RandomForestClassifier(), param_grid_rfc)
    }

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
    for name, (estimator, param_grid) in estimators_and_parameters.items():
        detailed_results_file = f'{name}_grid_search_detailed.csv'
        best_model, best_params, best_score = perform_grid_search(
            X_train, y_train, estimator, param_grid, cv_strategy, detailed_results_file)
        best_models[name] = best_model
        best_parameters[name] = best_params
        print(f"Best F1 score for {name}: {best_score}")
        print(f"Best parameters for {name}: {best_params}")

    # Evaluate each model
    for name, model in best_models.items():
        f1 = evaluate_model(model, X_test, y_test)
        print(f'{name} F1 Score on Test Set: {f1}')
