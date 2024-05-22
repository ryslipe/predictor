# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:36:23 2024

@author: ryans
"""
# dependencies
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_squared_error
import numpy as np
import pandas as pd


# import training data
df_train = pd.read_csv('data/train.csv')

# import testing data
df_test = pd.read_csv('data/test.csv')

# qb training data
qb_train = df_train.loc[df_train['position'] == 'QB'].copy()
qb_train_df = qb_train.to_csv('qb_training', index = False)

# quarterbacks data
# Let's make a copy of the qb_train dataframe that still has the **player_display_name**, **season**, and **week** columns. We want to do this so that we can grab these columns after the predictions for the model have been made.
quarterbacks_full = qb_train.copy()

# drop these columns
qb_dropping = ['position', 'recent_team', 'season',
       'week', 'opponent_team', 'completions', 'attempts', 'passing_yards',
       'passing_tds', 'interceptions', 'sack_fumbles_lost',
       'passing_first_downs', 'carries', 'rushing_yards', 'rushing_tds',
       'rushing_fumbles_lost', 'rushing_first_downs', 'receptions', 'targets',
       'receiving_yards', 'receiving_tds', 'receiving_fumbles_lost',
       'receiving_yards_after_catch', 'receiving_first_downs', 'target_share',
       'usage', 'comp_percentage', 'rb_def_avg', 'wr_def_avg', 'te_def_avg', 
       'last_twelve_receptions',
       'last_twelve_targets', 'last_twelve_receiving_yards',
       'last_twelve_receiving_tds', 'last_twelve_receiving_fumbles_lost',
       'last_twelve_receiving_yards_after_catch',
       'last_twelve_receiving_first_downs', 'last_twelve_target_share', 'last_twelve_usage']

### FUNCTION 6
def drop_columns(df, columns_to_drop):
    '''Drop the columns we will not be using for our analysis on quarterbacks along with na values.'''
    df.drop(columns = columns_to_drop, axis = 1, inplace = True)
    return 


# now we have only the columns that we need.
drop_columns(qb_train, qb_dropping)

# drop na values
qb_train.dropna(inplace = True)


# create X and y variables.
X_train_qb = qb_train.drop(columns = ['player_id', 'player_display_name', 'fantasy_points_ppr'], axis = 1)
y_train_qb = qb_train['fantasy_points_ppr'] 


# create our pipeline
pipelines = {
    'knn': make_pipeline(StandardScaler(), KNeighborsRegressor()),
    'rf' : make_pipeline(StandardScaler(), RandomForestRegressor()),
    'gb' : make_pipeline(StandardScaler(), GradientBoostingRegressor()),
    'ridge': make_pipeline(StandardScaler(), Ridge()),
    'lasso': make_pipeline(StandardScaler(), Lasso())
}

# these will be our 5 models. It will fit to our training data and create a dictionary of each model. 
# try to make that into a function.
def model_creation(X, y):
    dic = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X, y)
        dic[algo] = model
    return dic


# call our function to fit the models 
qb_mods = model_creation(X_train_qb, y_train_qb)


# function to see our RMSE on full training data (without cross validation being used.)
# rmse of our models on full training data - don't really need this tbh
def full_train_rmse(model_dict, X, y):
    '''A function to produce the RMSE on full training data without use of cross validation.'''
    rmse_models = {}
    for algo, model in model_dict.items():
        # make a prediction on training data using each model
        pred = model.predict(X)
        # calculate mse
        rmse = root_mean_squared_error(y, pred)
        # calculate rmse
        rmse_models[algo] = (rmse)
    return rmse_models

# call the function then print the results.
qb_train_rmse = full_train_rmse(qb_mods, X_train_qb, y_train_qb)
qb_train_rmse.items()

# neat way to print the results
def printing_rmse(rmse_dict):
    for k, v in rmse_dict.items():
        print(k,':',v)

# call printing function with our dictionary
printing_rmse(qb_train_rmse)

# bar graph of our rmse results
import matplotlib.pyplot as plt
x_val = ['knn', 'rf', 'gb', 'ridge', 'lasso']
y_val = list(qb_train_rmse.values())


# Graph the results (Make a better graph down the road but this works for now)
def make_rmse_plot(rmse_dict):
    x_val = ['knn', 'rf', 'gb', 'ridge', 'lasso']
    y_val = list(rmse_dict.values())
    # create the graph
    fig, ax = plt.subplots()
    ax.bar(x_val, y_val, color = ['Red', 'Green', 'Black', 'Orange', 'Blue'])
    ax.set_title('RMSE Bar Plot w/o CV', fontsize = 24)
    ax.set_ylabel('rmse', fontsize = 14)
    ax.set_ylim([0, 9])
    plt.show()
  
# call the plotting function
make_rmse_plot(qb_train_rmse)




#################################################################################
# Next, we get into the grid search. First, create the grid.
# grid 
grid = {
    'knn': {
        'kneighborsregressor__n_neighbors': [30, 35, 40, 45, 50],
    },
    'rf': {
        'randomforestregressor__n_estimators': [100, 150, 200],
        'randomforestregressor__max_features': [2, 3, 4, 5],
    },
    'gb':{
        'gradientboostingregressor__n_estimators': [100, 200, 300],
        'gradientboostingregressor__max_features': [2, 3, 4, 5],
        'gradientboostingregressor__learning_rate': [0.001, 0.01, 0.1, 0.5],
        'gradientboostingregressor__max_depth': [2, 3, 4]
    },
    'ridge':{
        'ridge__alpha': [20, 25, 30, 35],
    },
    'lasso': {
        'lasso__alpha': [0.25, 0.5, 0.75, 1]
    }
}


# fit the grid search for each model
def grid_search_models(position_model_dict, X, y):
    '''Function to perform gridsearch on all of our models.'''
    # empty dictionary to start
    searched_models = {}
    for algo, model in position_model_dict.items():
        # keep track of where its at
        print(f'Training the {algo} model')
        # find best parameters
        search = GridSearchCV(model, grid[algo], scoring = 'neg_mean_squared_error', cv = 5, return_train_score= True)
        # fit those best parameters on training data
        search.fit(X, y)
        # add to dictionary
        searched_models[algo] = search
    return searched_models

# call the function
qb_mods_cv = grid_search_models(qb_mods, X_train_qb, y_train_qb)


### ANOTHER FUNCTION (NUMBER IT)
def min_rmse(results):
    '''Function to return the lowest RMSE score of each model.'''
    # set up list for each cv score
    scores = []
    # find the 'neg_mean_squared_error' for each cv
    for mean_score in results['mean_test_score']:
        # get rmse by taking sqrt of neg mean_score
            scores.append(np.sqrt(-mean_score))
    # return lowest rmse
    return min(scores)

### FUNCTION FOR ALL CV RMSE
def cv_rmse(searched_model_dict):
    '''A function to get the lowest RMSE of our models.'''
    all_rmse = {}
    for algo, score in searched_model_dict.items():
        result = min_rmse(searched_model_dict[algo].cv_results_)
        all_rmse[algo] = result
    return all_rmse

# call function to create dictionary of all lowest RMSE of cross val models
qb_searched_rmse = cv_rmse(qb_mods_cv)

# call rmse printing function for neatly printed data.
printing_rmse(qb_searched_rmse)


# random forest model shows us feature importances
feature_importances_qb = qb_mods_cv['rf'].best_estimator_._final_estimator.feature_importances_

# set up the features for feature importance
attributes = list(X_train_qb.columns)
# importance of our features
sorted(zip(feature_importances_qb, attributes), reverse = True)

# make a function that gets feature importances and prints it 
def feature_importances(rf_model, X):
    '''Print the feature importances.'''
    feature_importances = rf_model.best_estimator_._final_estimator.feature_importances_
    # get columns to match feature importances
    attributes = list(X.columns)
    # importances
    importances = sorted(zip(feature_importances, attributes), reverse = True)
    return importances

# call the function for our importances
feature_importances(qb_mods_cv['rf'], X_train_qb)

# save each of the models for future use just in case. 
qb_knn = qb_mods_cv['knn']
qb_rf = qb_mods_cv['rf']
qb_gb = qb_mods_cv['gb']
qb_ridge = qb_mods_cv['ridge']
qb_lasso = qb_mods_cv['lasso']

# we may only be using one of the models but we can save them all incase we want to deploy them and let the user work with them. 
import joblib
joblib.dump(qb_knn, 'qb_knn_model.pkl')
joblib.dump(qb_rf, 'qb_rf_model.pkl')
joblib.dump(qb_gb, 'qb_gb_model.pkl')
joblib.dump(qb_ridge, 'qb_ridge_model.pkl')
joblib.dump(qb_lasso, 'qb_lasso_model.pkl')

#####################################################################################
# Testing Data
# create testing data
qb_test = df_test.loc[df_test['position'] == 'QB'].copy()

# keep name in a df
qb_test_full = qb_test.copy()

# call drop_qb_columns function for test data
drop_columns(qb_test, qb_dropping)

# drop na 
qb_test.dropna(inplace = True)

# create X and y variables.
X_test_qb = qb_test.drop(columns = ['player_id', 'player_display_name', 'fantasy_points_ppr'], axis = 1)
y_test_qb = qb_test['fantasy_points_ppr'] 

# we already have our trained models from before. Now we just have to test them.
def prediction_rmse(searched_model_dict, X, y):
    '''Make Predictions on Testing Data and print RMSE for each algorithm in our pipeline.'''
    for algo, model in searched_model_dict.items():
        pred = model.predict(X)
        mse = mean_squared_error(y, pred)
        rmse = np.sqrt(mse)
        print(f'Metrics for {algo}: {rmse}')

# call our function. The lasso model is the best model.         
prediction_rmse(qb_mods_cv, X_test_qb, y_test_qb)

# best model
final_qb_model = qb_mods_cv['lasso']
final_qb_predictions = final_qb_model.predict(X_test_qb)

# this part is from HOML book, make sure to site it!
from scipy import stats
confidence = 0.95
squared_errors = (final_qb_predictions - y_test_qb) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                        loc = squared_errors.mean(),
                        scale = stats.sem(squared_errors)))

# Quarterbacks Dataframe with predictions and actual
ypred = pd.Series(final_qb_predictions)
qb_name_final = qb_test_full['player_display_name'].tolist()
qb_week = qb_test_full['week'].tolist()
qb_points = qb_test_full['fantasy_points_ppr'].tolist()
qb_df_final = pd.DataFrame([qb_name_final, qb_week, qb_points, ypred]).T
qb_df_final.rename({0: 'Name', 1: 'Week', 2: 'Actual', 3: 'Predicted'}, axis = 1, inplace = True)
qb_df_final['Week'].unique()

qb_df_final.loc[qb_df_final['Week'] == 16]


# save to dataframe
qb_df_final.to_csv('qb_final_df', index = False)
