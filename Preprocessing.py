## Fetching data
# Connecting to database
import numpy as np
import sqlite3
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

###-----------col 1----------------
def get_match_label(match):
    ''' Derives a label for a given match. '''
    results = []
    for i in range(len(match)):

        home_goals = match['home_team_goal'].values[i]
        away_goals = match['away_team_goal'].values[i]
        label = pd.DataFrame()
        label.insert(0, 'match_api_id', match['match_api_id'])
        # Identify match label
        if home_goals > away_goals:
            results.insert(i, "Win")
        if home_goals == away_goals:
            results.insert(i, "Draw")
        if home_goals < away_goals:
            results.insert(i, "Defeat")
    match['class'] = results
    return label


def get_last_matches(matches, date, team, x=10):
    ''' Get the last x matches of a given team. '''

    # Filter team matches from matches
    team_matches = matches[(matches['home_team_api_id'] == team) | (matches['away_team_api_id'] == team)]
    # Filter x last matches from team matches
    last_matches = team_matches[team_matches.date < date].sort_values(by='date', ascending=False).iloc[0:x, :]

    # Return last matches
    return last_matches



def get_last_matches_against_eachother(matches, date, home_team, away_team, x=10):
    ''' Get the last x matches of two given teams. '''

    # Find matches of both teams
    home_matches = matches[(matches['home_team_api_id'] == home_team) & (matches['away_team_api_id'] == away_team)]
    away_matches = matches[(matches['home_team_api_id'] == away_team) & (matches['away_team_api_id'] == home_team)]
    total_matches = pd.concat([home_matches, away_matches])

    # Get last x matches
    try:
        last_matches = total_matches[total_matches.date < date].sort_values(by='date', ascending=False).iloc[
                       0:x, :]
    except:
        last_matches = total_matches[total_matches.date < date].sort_values(by='date', ascending=False).iloc[
                       0:total_matches.shape[0], :]

        # Check for error in data
        if (last_matches.shape[0] > x):
            print("Error in obtaining matches")

    # Return data
    return last_matches

###-----------col 4+5 ----------------
def create_class_column_5lastgames(match_data, df, name):
    scores = [];
    i = 0
    for x in df['match_api_id']:
        match_info = match_data.loc[match_data.match_api_id == x]
        matches_of_team = get_last_matches(match_data, match_info['date'].values[0],
                                           match_info[name].values[0], x=5)
        home_team = match_info[name].values[0]
        score_number = 0
        for y in matches_of_team['match_api_id']:
            match_info_results = df.loc[df.match_api_id == y]
            result = match_info_results['class'].values[0]
            if match_info_results['away_team_api_id'].values[0] == home_team:
                if result == "Defeat":
                    score_number = score_number + 3
            else:
                if result == "Win":
                    score_number = score_number + 3
            if result == "Draw":
                score_number = score_number + 1
        scores.insert(i, score_number / 15)
        i = i + 1

    df['5Last_Games' + name] = scores

def get_last_matches(matches, date, team, x=10):
    ''' Get the last x matches of a given team. '''
    # Filter team matches from matches
    team_matches = matches[(matches['home_team_api_id'] == team) | (matches['away_team_api_id'] == team)]
    # Filter x last matches from team matches
    last_matches = team_matches[team_matches.date < date].sort_values(by='date', ascending=False).iloc[0:x, :]

    # Return last matches
    return last_matches

###-----------col 6+7 ----------------
def create_class_5lastgames_between_teams(match_data, df, name):
    scores = [];
    i = 0
    for x in df['match_api_id']:
        match_info = match_data.loc[match_data.match_api_id == x]
        matches_of_team = get_last_matches_against_eachother(match_data, match_info['date'].values[0],
                                                             match_info['home_team_api_id'].values[0],
                                                             match_info['away_team_api_id'].values[0],
                                                             x=5)
        home_team = match_info[name].values[0]
        score_number = 0
        for y in matches_of_team['match_api_id']:
            match_info_results = df.loc[df.match_api_id == y]
            result = match_info_results['class'].values[0]
            if match_info_results['away_team_api_id'].values[0] == home_team:
                if result == "Defeat":
                    score_number = score_number + 3
            else:
                if result == "Win":
                    score_number = score_number + 3
            if result == "Draw":
                score_number = score_number + 1
        scores.insert(i, score_number / 15)
        i = i + 1

    df['five_last_meetings_for ' + name] = scores


###-----------col 8+9+10+11 ----------------
def get_fifa_stats(match, player_stats,df):
    ''' Aggregates fifa stats for a given match. '''

    # Define variables
    matchID = match.match_api_id
    date = match['date']
    playersHome = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11"]
    playersAway=["away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]
    rating = []

    # Loop through all players
    for player in playersHome:

        # Get player ID
        player_id = match[player]

        # Get player stats
        stats = player_stats[player_stats.player_api_id == player_id]

        if np.isnan(player_id) == True:
            overall_rating = 0
            rating.append(overall_rating)

        else:
            overall_rating=stats['overall_rating'].values
            rating.append(overall_rating[0])


    rating.sort(reverse=True)
    ratingForMainRating = rating[:5]
    avgForMainPlayers=np.average(ratingForMainRating)
    avgForAllPlayers=np.average(rating)

    df.loc[df['match_api_id']==matchID,'avg_performane_of_main_home_players']=avgForMainPlayers
    df.loc[df['match_api_id']==matchID,'avg_performane_of_all_home_players']=avgForAllPlayers

    rating = []
    # Loop through all players
    for player in playersAway:

        # Get player ID
        player_id = match[player]

        # Get player stats
        stats = player_stats[player_stats.player_api_id == player_id]

        if np.isnan(player_id) == True:
            overall_rating = 0
            rating.append(overall_rating)

        else:
            overall_rating=stats['overall_rating'].values
            rating.append(overall_rating[0])


    rating.sort(reverse=True)
    ratingForMainRating = rating[:5]
    avgForMainPlayers=np.average(ratingForMainRating)
    avgForAllPlayers=np.average(rating)

    df.loc[df['match_api_id']==matchID,'avg_performane_of_main_away_players']=avgForMainPlayers
    df.loc[df['match_api_id']==matchID,'avg_performane_of_all_away_players']=avgForAllPlayers


def ratio_week_per_games(early_date, late_date, num_games):
    delta = early_date - late_date
    x = delta.days / num_games
    ans = 7 / x
    return ans


###-----------col 12+13 ----------------
# average matches in a week for a match :
# every match get the 10 games before the games
def ave_match_in_week(df):
    dfDates=pd.DataFrame(columns=['match_api_id','team_away_id','team_home_id','average_game_per_week_home','average_game_per_week_away'])
    for game in df.itertuples():
        #get 10 last games for the team
         curr_date_match = datetime.datetime(int(game.date[0:4]), int(game.date[5:7]), int(game.date[8:10]))
         matches_of_home_team = get_last_matches(df, str(curr_date_match), game.home_team, x=10)
         if len(matches_of_home_team)>1  :
             early_date = matches_of_home_team.iloc[0]['date']
             early_date_datetime = datetime.datetime(int(early_date[0:4]), int(early_date[5:7]), int(early_date[8:10]))
             late_date = matches_of_home_team.iloc[len(matches_of_home_team)-1]['date']
             late_date_datetime = datetime.datetime(int(late_date[0:4]), int(late_date[5:7]), int(late_date[8:10]))
             # calculate the ratio  of average games in a week
             ratio_home = ratio_week_per_games(early_date_datetime,late_date_datetime,len(matches_of_home_team))
         else:
            ratio_home=0
         matches_of_away_team = get_last_matches(df, str(curr_date_match), game.away_team, x=10)
         if len(matches_of_away_team)>1:
             early_date = matches_of_away_team.iloc[0]['date']
             early_date_datetime = datetime.datetime(int(early_date[0:4]), int(early_date[5:7]), int(early_date[8:10]))
             late_date = matches_of_away_team.iloc[len(matches_of_away_team)-1]['date']
             late_date_datetime = datetime.datetime(int(late_date[0:4]), int(late_date[5:7]), int(late_date[8:10]))
             # calculate the ratio  of average games in a week
             ratio_away = ratio_week_per_games(early_date_datetime,late_date_datetime,len(matches_of_away_team))
         else:
            ratio_away=0
         new_row = {'match_api_id': game.match_api_id, 'team_home_id': game.home_team,'team_away_id': game.away_team,'average_game_per_week_home':ratio_home,'average_game_per_week_away':ratio_away}
         dfDates=dfDates.append(new_row,ignore_index=True)
    #assign mean in 0 value cells
    dfDates = dfDates.replace(0, np.NaN)
    dfDates['average_game_per_week_home'].fillna(dfDates['average_game_per_week_home'].mean(), inplace=True)
    dfDates['average_game_per_week_away'].fillna(dfDates['average_game_per_week_away'].mean(), inplace=True)
    print("ave_match_in_week done ")
    return dfDates



###-----------col 14+15 ----------------
def get_average_age_team(df_match,df_players):
    avaragedf=pd.DataFrame(columns=['match_api_id','team_away_id','team_home_id','average_age_home','average_age_away'])
    teams=[]
    # field to take from db
    players_fields_home = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
                      "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
                      "home_player_11"]
    players_fields_away = [ "away_player_1", "away_player_2", "away_player_3", "away_player_4",
                      "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
                      "away_player_10", "away_player_11"]
    for row in df_match.itertuples():
        players_home=[]
        players_away=[]
        curr_date_match=datetime.datetime(int(row.date[0:4]), int(row.date[5:7]), int(row.date[8:10]))
        #get last 10 matches of home team
        matches_of_home_team = get_last_matches(df_match, str(curr_date_match),row.home_team_api_id,x=5)
        for player in players_fields_home:
                i=0
                curr_players = matches_of_home_team[player].tolist()
                while i < len(curr_players):
                    if int(curr_players[i]) not in players_home:
                        players_home.append(int(curr_players[i]))
                    i=i+1
        #get last 10 matches of away team
        matches_of_away_team = get_last_matches(df_match, str(curr_date_match),row.away_team_api_id,x=5)
        for player in players_fields_away:
                i=0
                curr_players = matches_of_away_team[player].tolist()
                while i < len(curr_players):
                    if int(curr_players[i]) not in players_away:
                        players_away.append(int(curr_players[i]))
                    i=i+1
        sum=0
        counter=0
        #calculate the mean of the ages
        i=0
        while i < len(players_home):
            birth_date = df_players[df_players['player_api_id'] == players_home[i]]['birthday'].item()
            delta = curr_date_match - datetime.datetime(int(birth_date[0:4]), int(birth_date[5:7]), int(birth_date[8:10]))
            if(delta.days>0):
                counter=counter+1
                sum=sum+(delta.days/365)
            i=i+1
        if counter>0:
            average_home = sum/counter
        else:
            average_home=0
        sum_b=0
        counter=0
        i=0
        while i < len(players_away):
            birth_date = df_players[df_players['player_api_id'] == players_away[i]]['birthday'].item()
            delta = curr_date_match - datetime.datetime(int(birth_date[0:4]), int(birth_date[5:7]), int(birth_date[8:10]))
            if(delta.days>0):
                counter=counter+1
                sum_b = sum_b + (delta.days / 365)
            i=i+1
        if counter>0:
            average_away = sum_b /counter
        else:
            average_away=0
        players_away=[]
        players_home=[]
        new_row = {'match_api_id':row.match_api_id, 'team_home_id': row.home_team_api_id, 'team_away_id': row.away_team_api_id, 'average_age_home':average_home , 'average_age_away':average_away }
        avaragedf=avaragedf.append(new_row,ignore_index=True)
    #assign mean in 0 value cells
    avaragedf = avaragedf.replace(0, np.NaN)
    avaragedf['average_age_away'].fillna(avaragedf['average_age_away'].mean(), inplace=True)
    avaragedf['average_age_home'].fillna(avaragedf['average_age_home'].mean(), inplace=True)
    print("get_average_age_team done ")
    return avaragedf


###-----------col 12+13+14+15 ----------------
def add_values(df,df_avg_week,df_avg_age):
    for row in df_avg_week.itertuples():
        df.loc[df['match_api_id'] == row.match_api_id, 'home_team_avg_game_week'] = row.average_game_per_week_home
        df.loc[df['match_api_id'] == row.match_api_id, 'away_team_avg_game_week'] = row.average_game_per_week_away
    for row in df_avg_age.itertuples():
        df.loc[df['match_api_id'] == row.match_api_id, 'home_team_avg_age'] = row.average_age_home
        df.loc[df['match_api_id'] == row.match_api_id, 'away_team_avg_age'] = row.average_age_away
    return(df)

def get_ave_goal_for_home_team(matches, home_team,date):

    home_matches = get_last_matches(matches, date, home_team, x=10)
    home_goal_in_matches = home_matches["home_team_goal"].mean()
    return home_goal_in_matches

def get_ave_goal_for_away_team(matches, away_team,date):

    away_matches = get_last_matches(matches, date, away_team, x=10)
    away_goal_in_matches = away_matches["away_team_goal"].mean()
    return away_goal_in_matches

###-----------col 15+16----------------
def get_ave_goal_for_homeANDaway_team(match,df):
    match_data2 = match_data[['match_api_id', 'date']]
    dfemp = pd.merge(df, match_data2, how='left', on=['match_api_id'])
    for index, row in dfemp.iterrows():

        home_goal_in_matches = get_ave_goal_for_home_team(match, row['home_team_api_id'], row['date'])
        df.loc[df['home_team_api_id'] == row['home_team_api_id'], 'ave_goal_for_home_team'] = home_goal_in_matches

    for index, row in dfemp.iterrows():

        away_goal_in_matches = get_ave_goal_for_away_team(match, row['away_team_api_id'], row['date'])

        df.loc[df['away_team_api_id'] == row['away_team_api_id'], 'ave_goal_for_away_team'] = away_goal_in_matches



path = "C:/Users/shach/Desktop/Version5/"  #Insert path here
database = path + 'database.sqlite'
conn = sqlite3.connect(database)

d = {'home_team_api_id': [], 'away_team_api_id': [], 'home_team_goal': [], 'away_team_goal': [],
     'History_of_5last_games': [],
     'Result_against_for_teams': [], 'Home_game': [], 'ability_front_team': [], 'Average_of_players_age': [],
     'Injuried_main_players': [],
     'Injured_main_players': [], 'ave_match_in_week': [], 'Performance_of_main_players': [],
     'performance_of_all_players': [],
     'ave_goal_in_all_home': [], 'ave_goal_for_Home': []}

# Defining the number of jobs to be run in parallel during grid search
n_jobs = 1  # Insert number of parallel jobs here

# Fetching required data tables
player_data = pd.read_sql("SELECT * FROM Player;", conn)
player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
team_data = pd.read_sql("SELECT * FROM Team;", conn)
match_data = pd.read_sql("SELECT * FROM Match;", conn)

player_data = pd.read_sql("SELECT * FROM Player;", conn)
matchAge_data = pd.read_sql("SELECT * FROM Match;", conn)

# Reduce match data to fulfill run time requirements
rows = ["country_id", "league_id", "season", "stage", "date", "match_api_id", "home_team_api_id",
        "away_team_api_id", "home_team_goal", "away_team_goal", "home_player_1", "home_player_2",
        "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7",
        "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
        "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
        "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]

rows_player = ["id","player_api_id","player_name","player_fifa_api_id","birthday","height","weight"]
match_data.dropna(subset=rows, inplace=True)
player_data.dropna(subset=rows_player, inplace=True)
matchAge_data.dropna(subset=rows, inplace=True)

d_temp = {'match_api_id': match_data['match_api_id'].values}
df = pd.DataFrame(data=d_temp)


# ###-----------col 1----------------

# Creates target class Win/Defeat/Draw
###-----------col 2----------------
df.insert(1, "home_team_api_id", match_data['home_team_api_id'].values)

###-----------col 3----------------
df.insert(2, "away_team_api_id", match_data['away_team_api_id'].values)
df.insert(3, "home_team_goal", match_data['home_team_goal'].values)
df.insert(4, "away_team_goal", match_data['away_team_goal'].values)
get_match_label(df)

# Adding features
###-----------col 4----------------
create_class_column_5lastgames(match_data, df, 'away_team_api_id')
###-----------col 5----------------
create_class_column_5lastgames(match_data, df, 'home_team_api_id')
###-----------col 6----------------
create_class_5lastgames_between_teams(match_data, df, 'away_team_api_id')
###-----------col 7----------------
create_class_5lastgames_between_teams(match_data, df, 'home_team_api_id')


player_stats = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
player_stats = player_stats.groupby('player_api_id',as_index=False).mean()

match_data = pd.read_sql("SELECT * FROM Match;", conn)
###-----------col 8+9+10+11 ----------------
match_data.apply(lambda x :get_fifa_stats(x, player_stats,df),axis=1)


d_temp = {'match_api_id': match_data['match_api_id'].values,
          'home_team':match_data['home_team_api_id'].values,
          'away_team': match_data['away_team_api_id'].values,
          'date': match_data['date'].values,
          'home_team_api_id': match_data['home_team_api_id'].values,
          'away_team_api_id': match_data['home_team_api_id'].values}
df2 = pd.DataFrame(data=d_temp)


get_average_age_team(matchAge_data,player_data)
df = add_values(df,df_avg_week=ave_match_in_week(df2),df_avg_age=get_average_age_team(matchAge_data,player_data))

###-----------col 15+16----------------
get_ave_goal_for_homeANDaway_team(match_data,df)

print("standadization_and_normalize")
def standadization_and_normalize(data):
    # standardization data
    standartscalar = StandardScaler()
    # choose columns to normalize
    field = ['5Last_Gamesaway_team_api_id','5Last_Gameshome_team_api_id','five_last_meetings_for_away_team_api_id','five_last_meetings_for_home_team_api_id','avg_performance_of_main_home_players','avg_performance_of_all_home_players','avg_performance_of_main_away_players','avg_performance_of_all_away_players','home_team_avg_game_week','away_team_avg_game_week','home_team_avg_age','away_team_avg_age','ave_goal_for_home_team','ave_goal_for_away_team']
    #exectue the normalization
    df_scaled = pd.DataFrame(standartscalar.fit_transform(data.iloc[:, 3:17]), columns=field)
    df_scaled = pd.merge(data.iloc[:, :3], df_scaled, how='left', on= df_scaled.index)
    del df_scaled['key_0']
    df_scaled = pd.merge( df_scaled,data.iloc[:, 17:18], how='left', on= df_scaled.index)
    del df_scaled['key_0']
    print(df_scaled)
    return df_scaled
df= standadization_and_normalize(df)

df.to_csv("final.csv", index=False)
print("finish")