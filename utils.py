## random state
rst = 42

## for data
import numpy as np
import pandas as pd
import math
import time

## for pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')

## for plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

## for statistical tests
from statsmodels.stats import outliers_influence
from scipy import stats

## for machine learning
from sklearn import preprocessing, impute, utils, linear_model, feature_selection, model_selection, metrics, decomposition, cluster, ensemble, set_config, tree, neighbors, dummy
from imblearn import over_sampling, under_sampling, pipeline

#########################################################################################################################
#########################################################################################################################

# Outliers subplot
def style_axis(ax, ylabel, title=None):
    ax.set_ylabel(ylabel, fontsize=14)
    if title is not None:
        ax.set_title(title, fontsize=16)
    ax.grid(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    
def outliers_graph(df, cols, df2=None, col2=None, name1="Data 1", name2="Data 2"):
    sns.set_style("whitegrid")
    custom_palette = sns.color_palette("muted", max(len(cols), 2))
    ncols = min(len(cols), 4)
    nrows = math.ceil(len(cols) / ncols)

    if df2 is not None and col2 is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
        sns.boxplot(y=df[cols[0]], ax=ax1, color=custom_palette[0])
        sns.boxplot(y=df2[col2[0]], ax=ax2, color=custom_palette[1])

        style_axis(ax1, cols[0], f"{name1}")
        style_axis(ax2, col2[0], f"{name2}")

    else:
        # Plot for a single dataset
        if len(cols) == 1:
            fig, ax = plt.subplots(figsize=(4, 6))
            sns.boxplot(y=df[cols[0]], ax=ax, color=custom_palette[0])
            style_axis(ax, cols[0])
        else:
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
            ax = ax.flatten()

            for i in range(len(cols)):
                sns.boxplot(y=df[cols[i]], ax=ax[i], color=custom_palette[i])
                style_axis(ax[i], cols[i])

    for j in range(len(cols), nrows * ncols):
        fig.delaxes(ax[j])

    fig.tight_layout(w_pad=5.0)
    plt.show()

def target_dist(df, col_target, target_1_label, target_0_label):
    mpl.rcParams['font.size'] = 11
    r = df.groupby(col_target)[col_target].count()
    fig, ax = plt.subplots(figsize=(3,3))
    ax.pie(r, explode=[0.05, 0.1], labels=[target_1_label, target_0_label], radius=1.5, autopct='%1.1f%%', shadow=True, startangle=45,
           colors=['#66b3ff', '#ff9999'])
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    
def correlation_strength(corr):
    if corr >= 0.7 or corr <= -0.7:
        return 'Strong'
    elif 0.3 < corr < 0.7 or -0.7 < corr < -0.3:
        return 'Moderate'
    else:
        return 'Weak'

def select_features(X_train, y_train, X_test, k):
    fs = feature_selection.SelectKBest(score_func=feature_selection.f_classif, k=k)

    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    
    return X_train_fs, X_test_fs, fs, k

def evaluate_model(model, X_train, y_train, X_test, y_test, k=None):
    model.fit(X_train, y_train)
    
    y_pred = model.predict_proba(X_test)
    y_pred_pos_probs = y_pred[:, 1]
    
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_pos_probs)
    auc_score = metrics.auc(recall, precision)
    
    f1_score = (2 * precision * recall) / (precision + recall)
    ix_f1 = np.argmax(f1_score)
    
    f2_score = (5 * precision * recall) / (4 * precision + recall)
    ix_f2 = np.argmax(f2_score)
    
    print(f'{model.__class__.__name__}')
    print(f'PR AUC: {auc_score:.3f}')
    print(f'F1 Score={f1_score[ix_f1]:.3f}, Best Threshold={thresholds[ix_f1]:.3f}')
    print(f'F2 Score={f2_score[ix_f2]:.3f}, Best Threshold={thresholds[ix_f2]:.3f}')
    
    if k is not None:
        print(f'k={k}\n')

def pr_auc(y_test, y_pred_pos_probs):
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_pos_probs)
    return metrics.auc(recall, precision)

def run_grid_search(X_train, y_train):
    total_start_time = time.time()  # Start tracking the total time

    best_model = None
    best_params = None
    best_score = -np.inf
    best_clf_name = None

    for params in param_grid:
        if 'clf' in params:
            clf_name = str(params['clf'][0]).split('(')[0]  # Extract the model's name
        else:
            clf_name = "No Classifier (Feature Selection Step)"

        print(f"Starting grid search for model: {clf_name}")
        start_time = time.time()  # Start time tracking for individual model

        if params['clf'][0] == neighbors.KNeighborsClassifier():
            pipe_to_use = pipe_with_scaler
        else:
            pipe_to_use = pipe_without_scaler

        grid = model_selection.GridSearchCV(pipe_to_use, [params], cv=cv, scoring=metric, n_jobs=-1, verbose=2)
#         grid = model_selection.GridSearchCV(pipe_to_use, [params], cv=2, scoring=metric, n_jobs=-1, verbose=2)
        grid.fit(X_train, y_train)

        end_time = time.time()  # End time tracking for individual model
        elapsed_time = end_time - start_time

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_params = grid.best_params_
            best_model = grid.best_estimator_
            best_clf_name = clf_name

        print(f"\nModel: {clf_name}")
        print(f"Best parameters found: {grid.best_params_}")
        print(f"Best estimator: {grid.best_estimator_}")
        print(f"Best Score of Mean PR AUC: {grid.best_score_}")
        print(f"Time taken for {clf_name}: {elapsed_time:.2f} seconds\n")

    total_end_time = time.time()  # End tracking the total time
    total_elapsed_time = total_end_time - total_start_time

    print(f"\nTotal time taken for all models: {total_elapsed_time / 60:.2f} minutes")

    print("\n=== Best Model Found Across All Iterations ===")
    print(f"Best model: {best_clf_name}")
    print(f"Best estimator: {best_model}")
    print(f"Best parameters: {best_params}")
    print(f"Best Score of Mean PR AUC: {best_score}")
    
    return best_params
