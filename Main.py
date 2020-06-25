## Fetching data
# Connecting to database
import itertools
import sqlite3
import pandas as pd
import datetime
from sklearn.model_selection import KFold
import warnings
import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Loads the sql table
path = "C:\\Users\\pc\\Desktop\\"  #Insert path here
database = path + 'database.sqlite'
conn = sqlite3.connect(database)
match_data = pd.read_sql("SELECT * FROM Match;", conn)

df = pd.read_csv("final.csv")
d_temp = {'5Last_Gamesaway_team_api_id': [],
          "5Last_Gameshome_team_api_id": [],
          "last_meetings_for_away_team_api_id": [],
          "five_last_meetings_for_home_team_api_id": [],
          'avg_performance_of_main_home_players': [],
          'avg_performance_of_all_home_players': [],
          'avg_performance_of_main_away_players': [],
          "avg_performance_of_all_away_players": [],
          'home_team_avg_game_week': [],
          'away_team_avg_game_week': [],
          'home_team_avg_age': [],
          'away_team_avg_age': [],
          'ave_goal_for_home_team': [],
          'ave_goal_for_away_team': [],
          "class": []}
d_temp = {'match_id': match_data['match_api_id'].values}

# split the data fame to test and train by season 2015\2016 -test , other - train
df_Trn = pd.DataFrame(data=d_temp)
df_Tes = pd.DataFrame(data=d_temp)
print("Start")
match_data2 = match_data[['match_api_id', 'season']]
df = pd.merge(df, match_data2, how='left', on=['match_api_id'])

df_x = pd.DataFrame(data=d_temp)
df_y = df[['class']]

train = df[~df.season.isin(['2015/2016'])]
test = df[df.season.isin(['2015/2016'])]

X_tr = train[[
    #   '5Last_Gamesaway_team_api_id',
    #  '5Last_Gameshome_team_api_id',
    'five_last_meetings_for_away_team_api_id',
    'five_last_meetings_for_home_team_api_id',
    'avg_performance_of_main_home_players',
    'avg_performance_of_all_home_players',
    'avg_performance_of_main_away_players',
    'avg_performance_of_all_away_players',
    #      'home_team_avg_game_week',
    #       'away_team_avg_game_week',
    # 'home_team_avg_age',
    # 'away_team_avg_age',
    # 'ave_goal_for_home_team',
    # 'ave_goal_for_away_team'
]]
y_tr = train[['class']]
print(X_tr)
X_test = test[[
    #   '5Last_Gamesaway_team_api_id',
    #  '5Last_Gameshome_team_api_id',
    'five_last_meetings_for_away_team_api_id',
    'five_last_meetings_for_home_team_api_id',
    'avg_performance_of_main_home_players',
    'avg_performance_of_all_home_players',
    'avg_performance_of_main_away_players',
    'avg_performance_of_all_away_players',
    #      'home_team_avg_game_week',
    #       'away_team_avg_game_week',
    # 'home_team_avg_age',
    # 'away_team_avg_age',
    # 'ave_goal_for_home_team',
    # 'ave_goal_for_away_team'

]]
y_test = test[['class']]
print()
values = []

# Run several methods
print(df.groupby('class')['match_api_id'].nunique())
values.insert(0, 9810 / 21375)
print(9810 / 21375)

print("---------------------RandomForestClassifier----------------------------")
RF = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=3)
RF.fit(X_tr, y_tr.values.ravel())
RF.predict(X_test)
y_pred = RF.predict(X_test)
values.insert(1, accuracy_score(y_pred, y_test))
print(accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

plot_confusion_matrix(RF, X_test, y_test)
plt.title("RandomForestClassifier")

print("----------------------KNeighborsClassifier----------------------------")
KNN_model = KNeighborsClassifier(n_neighbors=330)
KNN_model.fit(X_tr, y_tr.values.ravel())
KNN_prediction = KNN_model.predict(X_test)
values.insert(2, accuracy_score(KNN_prediction, y_test))
print(accuracy_score(KNN_prediction, y_test))
print(classification_report(KNN_prediction, y_test))

plot_confusion_matrix(KNN_model, X_test, y_test)
plt.title("KNeighborsClassifier")

print("---------------------GaussianNB----------------------------")
gnb = GaussianNB()
y_pred = gnb.fit(X_tr, y_tr.values.ravel()).predict(X_test)
values.insert(3, accuracy_score(y_pred, y_test))
print(accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

plot_confusion_matrix(gnb, X_test, y_test)
plt.title("GaussianNB")

print("---------------------LogisticRegression----------------------------")
clf = LogisticRegression(random_state=5).fit(X_tr, y_tr.values.ravel())
y_pred = clf.predict(X_test)
values.insert(4, accuracy_score(y_pred, y_test))
print(accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

plot_confusion_matrix(clf, X_test, y_test)
plt.title("LogisticRegression")

print("---------------------AdaBoostClassifier----------------------------")
clf = AdaBoostClassifier(n_estimators=100, random_state=5)
y_pred = clf.fit(X_tr, y_tr.values.ravel()).predict(X_test)
values.insert(5, accuracy_score(y_pred, y_test))
print(accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

plot_confusion_matrix(clf, X_test, y_test)
plt.title("AdaBoostClassifier")

# Graphs

names = ['Apriori', 'RFC', 'KNN', 'NB', 'LR', 'AdaBoostC']
print(values)

plt.subplot(133)
plt.plot(names, values)
# plt.suptitle('Categorical Plotting')

for index, row in y_test.iterrows():
    if row['class'] == "Win":
        row['class'] = 0
    if row['class'] == "Defeat":
        row['class'] = 1
    if row['class'] == "Draw":
        row['class'] = 2

for i, label in enumerate(KNN_prediction):
    if label == "Win":
        KNN_prediction[i] = 0
    if label == "Defeat":
        KNN_prediction[i] = 1
    if label == "Draw":
        KNN_prediction[i] = 2
print(KNN_prediction)
arr = []
arr2 = []
for index, row in df.iterrows():
    arr.append(row['class'])

for i, label in enumerate(arr):
    if label == "Win":
        arr2.append(0)
    if label == "Defeat":
        arr2.append(1)
    if label == "Draw":
        arr2.append(2)
print(arr2)

fig = plt.figure(figsize=(6, 6))
a = fig.add_subplot(xlabel="avg_performance_of_all_home_players", ylabel="avg_performance_of_all_away_players")
a.scatter(X_test['avg_performance_of_all_home_players'], X_test['avg_performance_of_all_away_players'],
          c=y_test['class'])

fig2 = plt.figure(figsize=(6, 6))
b = fig2.add_subplot(xlabel="avg_performance_of_all_home_players", ylabel="avg_performance_of_all_away_players")
b.scatter(X_test['avg_performance_of_all_home_players'], X_test['avg_performance_of_all_away_players'],
          c=KNN_prediction)

fig3 = plt.figure(figsize=(6, 6))
a = fig3.add_subplot(xlabel="five_last_meetings_for_home_team_api_id", ylabel="avg_performance_of_all_home_players")
a.scatter(X_test['five_last_meetings_for_home_team_api_id'], X_test['avg_performance_of_all_home_players'],
          c=y_test['class'])

fig4 = plt.figure(figsize=(6, 6))
b = fig4.add_subplot(xlabel="five_last_meetings_for_home_team_api_id", ylabel="avg_performance_of_all_home_players")
b.scatter(X_test['five_last_meetings_for_home_team_api_id'], X_test['avg_performance_of_all_home_players'],
          c=KNN_prediction)

fig6 = plt.figure(figsize=(6, 6))
b = fig6.add_subplot(xlabel="ave_goal_for_home_team", ylabel="ave_goal_for_away_team")
b.scatter(df['ave_goal_for_home_team'], df['ave_goal_for_away_team'], c=arr2)

fig7 = plt.figure(figsize=(6, 6))
b = fig7.add_subplot(xlabel="home_team_avg_age", ylabel="away_team_avg_age")
b.scatter(df['home_team_avg_age'], df['away_team_avg_age'], c=arr2)


temp3 = pd.crosstab(X_test['five_last_meetings_for_home_team_api_id'], y_test['class'])
temp3.plot(kind='bar', stacked=True, color=['red','blue','black'], grid=False)

temp4 = pd.crosstab(df['ave_goal_for_home_team'], df['class'])
temp4.plot(kind='bar', stacked=True, color=['red','blue','black'], grid=False)

plt.show()
