# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 08:49:28 2024

@author: ryans
"""

# Program to import and preprocess our data
# import initial libraries
import pandas as pd

pd.set_option('display.max_columns', None)
import nfl_data_py as nfl


# get seasons 2020-2022
seasons = range(2020,2024)
df = nfl.import_weekly_data(seasons)

# remove the last week from the data
relevant = df['week'] < 18



# drop these columns
columns_to_drop = ['player_name', 'position_group', 'headshot_url', 'season_type',
       'sacks', 'sack_yards',
       'sack_fumbles', 'passing_air_yards',
       'passing_yards_after_catch', 'passing_epa',
       'passing_2pt_conversions', 'pacr', 'dakota', 'rushing_fumbles', 
       'rushing_epa', 'rushing_2pt_conversions',
       'receiving_fumbles', 'receiving_air_yards', 'receiving_epa',
       'receiving_2pt_conversions', 'racr', 'air_yards_share',
       'wopr', 'special_teams_tds', 'fantasy_points']




## FUNCTION NUMBER 1 
def initial_drops(df):
    '''Drop columns and rows we will not be examining.'''
    # not using week 18
    relevant = df['week'] < 18
    
    # boolean indexing
    df = df[relevant]
    
    # establish columns to be dropped
    columns = columns_to_drop
    df.drop(columns = columns, axis = 1, inplace = True)
    
    return df



    
# call the function on our dataframe
df_new = initial_drops(df)

# check weeks
df_new['week'].unique

# create completion percentage and usage 
df_new['comp_percentage'] = df_new['completions'] / df_new['attempts']
df_new['usage'] = df_new['carries'] + df_new['targets']

# make sure it is sorted by player and game
df_new.sort_values(by = ['player_id', 'season', 'week'])


# get defensive points allowed by using grouping
group_df = df_new.groupby(['opponent_team', 'season', 'week', 'position'])['fantasy_points_ppr'].mean().reset_index(name = 'def_fantasy_points')


# dont forget to modularize this part
# try to split by position and team then merge again
qb_def = group_df.loc[group_df['position'] == 'QB']
rb_def = group_df.loc[group_df['position'] == 'RB']
wr_def = group_df.loc[group_df['position'] == 'WR']
te_def = group_df.loc[group_df['position'] == 'TE']

### FUNCTION NUMBER 6
def rolling_avg(df, window):
    '''A function to compute the rolling averages of our statistics.'''
    # shift by one so we do not include this weeks unknown statistics
    return df.shift(1).rolling(window = window, min_periods = 1).mean()

# we are going to fill the NA values with the NFL average defensive fantasy points allowed
# get rolling averages for them - shift one place
qb_def['rolling_def'] = qb_def.groupby('opponent_team')['def_fantasy_points'].apply(lambda x: rolling_avg_try(x, 12)).reset_index(0, drop = True)

rb_def['rolling_def'] = rb_def.groupby('opponent_team')['def_fantasy_points'].apply(lambda x: rolling_avg_try(x, 12)).reset_index(0, drop = True)

wr_def['rolling_def'] = wr_def.groupby('opponent_team')['def_fantasy_points'].apply(lambda x: rolling_avg_try(x, 12)).reset_index(0, drop = True)

te_def['rolling_def'] = te_def.groupby('opponent_team')['def_fantasy_points'].apply(lambda x: rolling_avg_try(x, 12)).reset_index(0, drop = True)

# merge them all 
dataframes = [qb_def, rb_def, wr_def, te_def]

from functools import reduce
defense = reduce(lambda left, right: pd.merge(left, right, on = ['opponent_team', 'season', 'week', 'position', 'def_fantasy_points', 'rolling_def'], how = 'outer'), dataframes)

# filter down to players who have only scored over 150 points
df_new = df_new[df_new.groupby('player_id')['fantasy_points_ppr'].transform('sum') > 150]

# merge defensive dataframe with our offensive dataframe
merged_df = df_new.merge(defense, how = 'inner', on = ['opponent_team', 'season', 'week', 'position'])




#####################################################################################################
merged_df = merged_df.sort_values(by = ['player_id', 'season', 'week'])
full_df = merged_df.copy()



# stats that we want rolling averages for 
avgs = ['completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions', 'sack_fumbles_lost', 'passing_first_downs', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles_lost', 'rushing_first_downs', 'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_fumbles_lost', 'receiving_yards_after_catch', 'receiving_first_downs', 'target_share', 'fantasy_points_ppr', 'usage', 'comp_percentage']


# Heres the deal. I think we are gonna have to go in to rosters then merge with our dataframe then run the rolling averages then split the first week to testing then run the pipeline and train the model.The test data will be that last week. This should only have to happen for the first week of every season. 

# get rosters
rosters = nfl.import_seasonal_rosters(years = range(2024, 2025)) 

# Next up we gotta clean the rosters. Get rid of positions we dont use, retired players. Then we gotta get the schedules. 
# only these positions
positions = ['QB', 'RB', 'WR', 'TE']

# only these positions
rosters = rosters.loc[rosters['position'].isin(positions)]

# only active players
rosters = rosters.loc[rosters['status'] == 'ACT']

# only these columns
rosters = rosters[['player_id', 'player_name', 'position', 'team', 'season', 'week']]

rosters.rename(columns = {'player_name': 'player_display_name', 'team': 'recent_team'}, inplace = True)

# get schedules
# import schedule for new unseen data
sched = nfl.import_schedules(range(2024, 2025))

# only these columns
sched = sched[['season', 'week', 'home_team', 'away_team']]

sched.rename(columns = {'home_team': 'team'}, inplace = True)
sched = sched.loc[sched['week'] == 1]


# this part is weird we are basically just adding the opponent team to the rosters df but a few steps must be taken. 
team_home_mapping = dict(zip(sched['team'], sched['away_team']))
team_away_mapping = dict(zip(sched['away_team'], sched['team']))

go = pd.DataFrame.from_dict(team_home_mapping, orient = 'index')
go.reset_index(inplace = True)
go.rename(columns = {'index': 'recent_team', 0: 'away_team'}, inplace = True)

go2 = pd.DataFrame.from_dict(team_away_mapping, orient = 'index')
go2.reset_index(inplace = True)
go2.rename(columns = {'index': 'recent_team', 0: 'away_team'}, inplace = True)

# these are the values we need for our dataframe!
con = pd.concat([go, go2], ignore_index = True)

# players and opponents 
df_24 = rosters.merge(con, on = 'recent_team', how = 'outer')

# rename away team to opponent team
df_24.rename(columns = {'away_team': 'opponent_team'}, inplace = True)





### FUNCTION NUMBER 7
def statistic_avgs_try(df, col):
    '''Get rolling averages for specific columns.'''
    for col in df[col]:
        df[f'last_twelve_{col}'] = df.groupby('player_id')[col].apply(lambda x: rolling_avg_try(x, 12)).reset_index(0, drop = True)
    return df


### FUNCTION NUMBER 6
def rolling_avg_try(df, window):
    '''A function to compute the rolling averages of our statistics.'''
    # shift by one so we do not include this weeks unknown statistics
    return df.rolling(window = window, min_periods = 1).mean()

# dataframe with defense but no rolling averages
un_shift = merged_df.copy()

# adds 2024 week 1 players and opponents
no_shift = un_shift.merge(df_24, on = ['player_id', 'player_display_name', 'position', 'recent_team', 'season', 'opponent_team', 'week'], how = 'outer')

# take that dataframe and compute rolling averages with NO SHIFT
no_shift.sort_values(by = ['player_id', 'season', 'week'], inplace = True)


no_shift = statistic_avgs_try(no_shift, avgs)

# try to make a function to shift only the avgs columns in our dataframe 
final = no_shift.copy()


# columns to be shifted 
last_twelve = ['last_twelve_completions', 'last_twelve_attempts',
'last_twelve_passing_yards', 'last_twelve_passing_tds',
'last_twelve_interceptions', 'last_twelve_sack_fumbles_lost',
'last_twelve_passing_first_downs', 'last_twelve_carries',
'last_twelve_rushing_yards', 'last_twelve_rushing_tds',
'last_twelve_rushing_fumbles_lost', 'last_twelve_rushing_first_downs',
'last_twelve_receptions', 'last_twelve_targets',
'last_twelve_receiving_yards', 'last_twelve_receiving_tds',
'last_twelve_receiving_fumbles_lost',
'last_twelve_receiving_yards_after_catch',
'last_twelve_receiving_first_downs', 'last_twelve_target_share',
'last_twelve_fantasy_points_ppr', 'last_twelve_usage',
'last_twelve_comp_percentage', 'rolling_def']

# function to shift them 
def shifting(df, col):
    for col in df[col]:
        df[col] = df.groupby('player_id')[col].shift(1)
    return df

# rolling averages fixed for 2024
final = shifting(final, last_twelve)

############################################################################################################

###############################################################################
# Now its time to create our training and testing data. The testing data will be the last 4 weeks of the 2023 season. This will be accomplished using boolean indexing.

# create a copy of data
splits = final.copy()


# create series that is 2022 season
last_season = splits['season'] == 2023

# create series that is last 4 games
last_four = splits['week'] >= 14

# last 4 weeks of the last season for test data
combined = last_season & last_four

# test data is combined
test = splits[combined]

# everything but testing data
not_testing = ~combined

# training data
train = splits[not_testing]

# save train and test as pd
train.to_csv('data/train_new.csv', index = False)
test.to_csv('data/test_new.csv', index = False)


#################################################################################################################
# use boolean masking to create training and testing data
first_game = final['season'] == 2024

# testing data 
full_test = final[first_game]

# not 2024 week 1
full_train = ~first_game

# our full training data
full_training = final[full_train]


############################################################################################################
# Remember, there is no RMSE for our predictions because we do not know the actual fantasy points scored. 

# save both to csv
full_training.to_csv('data/full_training.csv', index = False)
full_test.to_csv('data/full_test.to_csv')


