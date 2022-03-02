import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

data = pd.read_csv('D:/Data/streaming_history.csv')
history = pd.read_csv('D:/Data/history_raw.csv')

history['seconds'] = history['msPlayed'] / 1000
history['min'] = history['seconds'] / 60

artist = history.groupby(['artistName']).sum()
track = history.groupby(['trackName']).sum()

from datetime import datetime
history['endTime'] = history['endTime'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M'))

newhistory = history[history['endTime'] > '05-01-2020']
artist_new = newhistory.groupby(['artistName']).sum()
track_new = newhistory.groupby(['trackName']).sum()

comb = pd.merge(data, track, how='left', left_on='name', right_on='trackName')
comb = comb[['name', 'key', 'danceability', 'energy', 'loudness', 'speechiness',
             'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
             'msPlayed', 'seconds', 'min']]

#comb.to_csv('D:/Data/trackFeatures.csv')

comb = pd.read_csv('D:/Data/trackFeatures.csv')

f,ax=plt.subplots(1,2,figsize=(18,8))
comb['bitLike'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%', ax=ax[0],
                                     shadow=True)
ax[0].set_title('Liked')
ax[0].set_ylabel('')
sns.countplot('bitLike', data=comb, ax=ax[1])
ax[1].set_title('Liked')
plt.show()

numerical = ['key', 'danceability', 'energy', 'loudness', 'speechiness',
             'acousticness', 'instrumentalness', 'liveness', 'valence',
             'tempo', 'msPlayed']
comb[numerical].hist(bins=15, figsize=(15,6), layout=(3,4));

def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(3,4,figsize=(18,8))

    for feature in features:
        i += 1
        plt.subplot(3,4,i)
        sns.kdeplot(df1[feature], bw=0.5,label=label1)
        sns.kdeplot(df2[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=11)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show();
t0 = comb.loc[comb['bitLike'] == 0]
t1 = comb.loc[comb['bitLike'] == 1]
plot_feature_distribution(t0, t1, 'bitLike: 0', 'bitLike: 1', numerical)

target = comb['bitLike']
features = [c for c in comb.columns if c not in ['name', 'msPlayed',
                                                 'seconds', 'min', 'bitLike']]
param = {
    'boost_from_average':'false',
    'boost': 'gbdt',
    'learning_rate': 0.01,
    'max_depth': -1,
    'metric': 'auc',
    'objective': 'binary',
    'verbosity': 1
    }

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)
oof = np.zeros(len(comb))
predictions = np.zeros(len(comb))
feature_importance = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(comb.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(comb.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(comb.iloc[val_idx][features], label=target.iloc[val_idx])
    
    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100,
                    early_stopping_rounds = 500)
    oof[val_idx] = clf.predict(comb.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance = pd.concat([feature_importance, fold_importance_df], axis=0)
    
print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))