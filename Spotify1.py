import ast
from typing import List
from os import listdir

def get_streamings(path: str = 'MyData') -> List[dict]:
    
    files = ['MyData/' + x for x in listdir(path)
             if x.split('.')[0][:-1] == 'StreamingHistory']
    
    all_streamings = []
    
    for file in files:
        with open(file, 'r', encoding='UTF-8') as f:
            new_streamings = ast.literal_eval(f.read())
            all_streamings += [streaming for streaming
                               in new_streamings]
    return all_streamings


import spotipy.util as util

username = ''
client_id = ''
client_secret = ''
redirect_uri = ''
scope = 'user-read-recently-played'

token = util.prompt_for_user_token(username=username,
                                   client_id=client_id,
                                   client_secret=client_secret,
                                   redirect_uri=redirect_uri)


print(token)

import requests
def get_id(track_name: str, token: str) -> str:
    headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer ' + token,
    }
    params = [
    ('q', track_name),
    ('type', 'track'),
    ]
    try:
        response = requests.get('https://api.spotify.com/v1/search',
                                headers = headers, params = params, timeout = 5)
        json = response.json()
        first_result = json['tracks']['items'][0]
        track_id = first_result['id']
        return track_id
    except:
        return None
    
import spotipy
def get_features(track_id: str, token:str) -> dict:
    sp = spotipy.Spotify(auth=token)
    try:
        features = sp.audio_features([track_id])
        return features[0]
    except:
        return None
    
streamings = get_streamings()
unique_tracks = list(set([streaming['trackName']
                          for streaming in streamings]))


all_features = {}
for track in unique_tracks:
    track_id = get_id(track, token)
    features = get_features(track_id, token)
    if features:
        all_features[track] = features
        
with_features = []
for track_name, features in all_features.items():
    with_features.append({'name': track_name, **features})
    
import pandas as pd
df = pd.DataFrame(with_features)
df.to_csv('./streaming_history.csv')

history = pd.DataFrame(streamings)
history['seconds'] = history['msPlayed'] / 1000
history['min'] = history['seconds'] / 60
history.to_csv('./history_raw.csv')

artist = history.groupby(['artistName']).sum()
track = history.groupby(['trackName']).sum()

from datetime import datetime
history['endTime'] = history['endTime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M'))

newhistory = history[history['endTime'] > '05-01-2020']
artist_new = newhistory.groupby(['artistName']).sum()
track_new = newhistory.groupby(['trackName']).sum()


df = df.iloc[:,:12]
