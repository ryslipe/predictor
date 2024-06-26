# -*- coding: utf-8 -*-
"""
Created on Mon May 20 19:17:48 2024

@author: ryans
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_squared_error
from streamlit_extras.no_default_selectbox import selectbox

# best models 
# te = ridge
# wr = ridge
# rb = ridge
# qb = lasso

quarterbacks_full = pd.read_csv('data/quarterbacks_train_new')
df = pd.read_csv('data/qb_final_df_new')
df.rename(columns = {'Predicted': 'predicted'}, inplace = True)
qb_train = pd.read_csv('data/qb_training_new')

df_table = df.copy()
df_table['season'] = df_table['season'].astype(str)

with st.sidebar:
    selected = option_menu(
        menu_title = 'Main Menu',
        options = ['Quarterbacks', 'Runningbacks', 'Wide Receivers', 'Tight Ends', 'User Guide'],
        default_index = 0
        )

if selected == 'Quarterbacks':    

    # title of our app
    st.title('Predicting Fantasy Points :football:')
    
    # introductory paragraph
    st.write('Welcome to the Fantasy Football Machine Learning Predictor! In this first phase of rollouts, we are dealing with only quarterbacks. The data consists of training data fro the 2020, 2021, and first 13 weeks of the 2022 seasons. The model is then tested on the last 4 games of the 2022 season. Each season had the final game removed from the data because it is not representative of the population. In the final week of the season many teams rest their best players or play them in small amounts to avoid injury. We do not want this week to disturb the statistics used for prediction. The model uses a 12 weeek rolling average of various player statistics to come up with a prediction. For quarterbacks, a "lasso" model gave the lowest RMSE. It is tested on the last four weeks because this is generally the time frame of fantasy football playoff matchups.')
    
    # first section - player predictions
    st.header('Player Predictions')
    # explain the search bar
    st.write('To view the results of the model enter a player that you would like to see predictions for. If the player has no data it means they did not play during the final 4 games of the season. The sortable table includes the player name along with the week and actual and predicted points scored. Click on the column to sort the predictions.')
   
    
   # enter a player name to display predictions
    text_search = st.text_input('Enter a player name. If table is empty, player not found.', '')
    # m1 = df["Name"].str.contains(text_search.title())
    
    # function to create table
    def make_table(text_search):
        table = df['player_display_name'].str.contains(text_search.title())
        return table
    
    table = make_table(text_search)
    
    if text_search:
        searched_table = df[table]
        searched_table['season'] = searched_table['season'].astype(str).str.replace(',', '')
        st.write(searched_table)
    
    # dataframe downloader
    @st.cache_data
    def df_converter(df):
        return df.to_csv().encode('utf-8')
    csv = df_converter(df)
    
    st.write('\U0001F447 To see every quarterback''s predictions download the dataset here. \U0001F447')
    st.download_button(
     label="Download data as CSV",
     data=csv,
     file_name='qb_projections_df.csv',
     mime='text/csv',
 )
    
    
    st.header('Week by Week Predictions')
    st.write('Choose a week to display the predictions of every quarterback for the selected week.')
    # choose a week to display 
    text_2 = st.select_slider('Choose a Week Number', [14, 15, 16, 17])
    
    if text_2:
        df['season'] = df['season'].astype(str).str.replace(',', '')
        df.loc[df['week'] == text_2]
        
    
    # function to make graph of comparisons
    def compare(player_1, player_2):
        '''A function to graph comparision.'''
        first_line = df.loc[df['player_display_name'] == player_1]
        second_line = df.loc[df['player_display_name'] == player_2]
        
        # graph them
        fig, ax = plt.subplots()
        ax.plot(first_line['week'], first_line['predicted'], label = player_1, marker = 'o')
        ax.plot(second_line['week'], second_line['predicted'], label = player_2, marker = 'o')
        plt.xticks([14, 15, 16, 17])
        plt.title(f"Comparison of {player_1} and {player_2}")
        plt.xlabel('Week')
        plt.ylabel('Fantasy Points')
        plt.legend()
        return fig
    
    # next section - graphical comparison
    st.header('Graphical Comparison')
    st.write('To make comparisons of two players easy to interpret, enter two players for a line graph of the predicted points for the final 4 games of the 2022 season. ')
    # input for player 1 and 2
    player_1 = st.text_input('Enter First Player', '').title()
    player_2 = st.text_input('Enter Second Player', '').title()
    
    if player_1 and player_2:
        fig = compare(player_1, player_2)
        st.pyplot(fig)
        
    
    
    def who_to_start(week, player_1, player_2):
        '''A function to decide which player should start.'''
        # subset of dataframe
        player_1_name = df.loc[(df['player_display_name'] == player_1) & (df['week'] == week)]
        player_1_points = player_1_name['predicted'].tolist()
        player_2_name = df.loc[(df['player_display_name'] == player_2) & (df['week'] == week)]
        player_2_points = player_2_name['predicted'].tolist()
        if player_1_points and player_2_points:
        
            # names
            names = [player_1, player_2]
            # points
            points = [player_1_points, player_2_points]
            # zip them
            most_points = max(points)
            # who to start
            starter = points.index(most_points)
            best_player = names[starter]
            st.write(f'Start {best_player}')
            st.write('Player Predictions:')
            st.write(f'{player_1}: {player_1_points}')
            st.write(f'{player_2}: {player_2_points}')
        else:
            st.write(f'Please Choose Two Players who are starting for week {week}.')
    
        # next section - who to start
    st.header('Who to Start')  
    # explain the "who to start" function
    st.write('Do you have two players that you are unsure about starting? These tough decisions could be costly. Let the model make the decision for you. Type in the week you want along with the two players you are deciding between and the model will tell you who you should start. If the player entered is not playing in those weeks you will be asked to try again.')  
    # input for player 1 and 2
    week_starter = st.selectbox('Pick a week for starting comparison', [14, 15, 16, 17])
    player_starter_1 = st.text_input('Enter a player to start')
    player_starter_2 = st.text_input('Enter a second player to start')
    
    if (week_starter) and (player_starter_1) and (player_starter_2):
    
        who_to_start(int(week_starter), player_starter_1, player_starter_2)
  


    # using the model to make it work
    quarterbacks_full = qb_train.copy()
    
    
    # drop na values
    qb_train.dropna(inplace = True)
    
    
    # create X and y variables.
    X_train_qb = qb_train.drop(columns = ['player_id', 'player_display_name', 'fantasy_points_ppr'], axis = 1)
    y_train_qb = qb_train['fantasy_points_ppr'] 
    
    ######################################################################################################################################################
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
    
    
    # bar graph of our rmse results
    import matplotlib.pyplot as plt
    # set up x and y values
    x_val = ['knn', 'rf', 'gb', 'ridge', 'lasso']
    y_val = list(qb_train_rmse.values())
    
    
    # Graph the results 
    def make_rmse_plot(rmse_dict, title, ylim):
        x_val = ['knn', 'rf', 'gb', 'ridge', 'lasso']
        y_val = list(rmse_dict.values())
        # create the graph
        fig_1, ax = plt.subplots()
        ax.bar(x_val, y_val, color = ['Red', 'Green', 'Black', 'Orange', 'Blue'])
        ax.set_title(title, fontsize = 24)
        ax.set_ylabel('rmse', fontsize = 14)
        ax.set_ylim(ylim)
        return fig_1
    
    # call the plotting function
    fig_1 = make_rmse_plot(qb_train_rmse, 'RMSE Plot without Cross Validation', [0, 9])
    if st.button('Generate RMSE Report'):
        st.pyplot(fig_1)
        
    st.write('The results of the RMSE show that random forest is the best model but there is potenial for overfitting.')
    
    
    
    
    # random forest model shows us feature importances
    importances = pd.read_csv('data/importances.csv')
    st.write(importances)
    ######################################################################################################################################################
    # call the plotting function
    cv_rmse_dict = {'knn': 7.995205302511437,
     'rf': 7.920867882577945,
     'gb': 7.942460224774128,
     'ridge': 7.88654632047547,
     'lasso': 7.880494637687264}
    
    fig_2 = make_rmse_plot(cv_rmse_dict, 'Graph of Cross Validation RMSE', [7, 9])
    if st.button('Generate Grid Searched RMSE Report'):
        st.pyplot(fig_2)
        
    st.write('The results of the plot show the RMSE values got higher but not by too much. The lowest RMSE is from the Lasso model but they are all very close. This is the reason the model chosen was the Lasso model. In future rollouts, I will implement an ensemble of methods along with neural networks and time series analysis techniques.')
    
    
    st.header('Descriptive Statistics')
    st.write('The descriptive statistics are displayed below. Since the range of values are much different it is imoprtant to scale the data for the Lasso model.')
    st.write(qb_train.describe().T)
    
    st.write('One of the interesting parts of the data analysis is to look at the correlation of our features with our target variable. None of these are extremely correlated to the target alone, but with interactions among other variables, our predictions are quite accurate for most players. ')
    corr_matrix = qb_train.iloc[:, 2:].corr()
    st.write(corr_matrix['fantasy_points_ppr'].sort_values(ascending = False))
    
    st.write('We can get a graph of our players actual points from the training data along with the projected points from the testing data to see how they are trending.')
    
    
    
    # graph of the players training data along with testing data
    qb_df = qb_train.copy()
    player = set(qb_train['player_display_name'])
    st.header('Projection Overlay')
    st.write('Choose a player from the drop down menu to see their historical points graphed in black and their projections graphed in red. If there is no red line it means the player did not play in the final four weeks of the 2022 season.')
    full_player = selectbox('Pick a player from the drop down menu.', player)
    choice = full_player
    master_set = pd.concat([quarterbacks_full, df], axis = 0, ignore_index = True)
    master_set['period'] = master_set['season'].astype(str) + master_set['week'].astype(str)
    st.write(master_set['period'].unique())
    
    df_final = df.copy()
    
    def full_graph(player, master_set):
        '''Function to graph a player's actual from training and projected from testing.'''
        # df of player 
        actual = master_set.loc[master_set['player_display_name'] == player]
        # reset index 
        actual.reset_index(inplace = True)
        # add index column
        actual['index'] = actual.index
        # points
        y_vals = actual['fantasy_points_ppr']
        fig3, ax = plt.subplots()
        
        test_projections = actual['predicted']
        ax.plot(actual['period'], y_vals, color = 'black', marker = 'o', label = 'Actual Points')
        ax.plot(actual['period'], test_projections, color = 'red', marker = 'o', label = 'Predicted Points')
        ax.set_title(f'Historic Points with Projection Overlay for {player}')
        ax.set_ylabel('Fantasy Points')
        ax.grid(True)
        ax.legend()
        return fig3
        
    
    
    #if choice:
        #fig3 = full_graph(choice, master_set)
        #st.pyplot(fig3)
        
    
    
if selected == 'Runningbacks':
    st.title(f'{selected} Coming Soon')
if selected == 'Wide Receivers':
    st.title(f'{selected} Coming Soon')
if selected == 'Tight Ends':
    st.title(f'{selected} Coming Soon')
if selected == 'User Guide':
    st.title(f'{selected}')
    st.write('Welcome to the user guide for the Fantasy Football Machine Learning Predictor.')
