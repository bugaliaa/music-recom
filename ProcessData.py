import pandas as pd
import numpy as np
import json
import re 
import sys
import itertools

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util


spotify_df = pd.read_csv(r'C:\Users\Bugalia\musicRec\tracks.csv')
data_w_genres = pd.read_csv(r'C:\Users\Bugalia\musicRec\artists.csv')

def process():
    data_w_genres['genres_upd'] = data_w_genres['genres'].apply(lambda x: [re.sub(' ','_',i) for i in re.findall(r"'([^']*)'", x)])
    
    spotify_df['artists_upd_v1'] = spotify_df['artists'].apply(lambda x: re.findall(r"'([^']*)'", x))
    spotify_df['artists_upd_v2'] = spotify_df['artists'].apply(lambda x: re.findall('\"(.*?)\"',x))
    spotify_df['artists_upd'] = np.where(spotify_df['artists_upd_v1'].apply(lambda x: not x), spotify_df['artists_upd_v2'], spotify_df['artists_upd_v1'] )

    spotify_df['artists_song'] = spotify_df.apply(lambda row: str(row['artists_upd'][0])+str(row['name']), axis = 1)

    spotify_df.sort_values(['artists_song','release_date'], ascending = False, inplace = True)

    spotify_df.drop_duplicates('artists_song',inplace = True)

    artists_exploded = spotify_df[['artists_upd','id']].explode('artists_upd')

    artists_exploded_enriched = artists_exploded.merge(data_w_genres, how = 'left', left_on = 'artists_upd',right_on = 'name')
    artists_exploded_enriched_nonnull = artists_exploded_enriched[~artists_exploded_enriched.genres_upd.isnull()]

    artists_genres_consolidated = artists_exploded_enriched_nonnull.groupby('id_x')['genres_upd'].apply(list).reset_index()
    artists_genres_consolidated['consolidates_genre_lists'] = artists_genres_consolidated['genres_upd'].apply(lambda x: list(set(list(itertools.chain.from_iterable(x)))))

    artists_genres_consolidated.rename(columns = {'id_x':'id'}, inplace = True)

    spotify_df = spotify_df.merge(artists_genres_consolidated[['id','consolidates_genre_lists']], on = 'id', how = 'left')

    spotify_df['year'] = spotify_df['release_date'].apply(lambda x: x.split('-')[0])

    spotify_df['popularity_red'] = spotify_df['popularity'].apply(lambda x: int(x/5))
    spotify_df.rename(columns = {'consolidates_genre_lists_x':'consolidates_genre_lists'}, inplace = True)

    spotify_df['consolidates_genre_lists'] = spotify_df['consolidates_genre_lists'].apply(lambda d: d if isinstance(d, list) else [])


def ohe_prep(df, column, new_name):
    tf_df = pd.get_dummies(df[column])
    features_names  = tf_df.columns
    tf_df.columns = [new_name + '|' + str(col) for col in features_names]
    tf_df.reset_index(drop = True, inplace = True)
    return tf_df

def create_feature_set(df, float_cols):
    tfidf = TfidfVectorizer()
    tfidf_matrix =  tfidf.fit_transform(df['consolidates_genre_lists'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names()]
    genre_df.reset_index(drop = True, inplace=True)

    year_ohe = ohe_prep(df, 'year','year') * 0.5
    popularity_ohe = ohe_prep(df, 'popularity_red','pop') * 0.15

    floats = df[float_cols].reset_index(drop = True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2

    final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis = 1)

    final['id'] = df['id'].values

    return final

def main():
    process()
    float_cols = spotify_df.dtypes[spotify_df.dtypes == 'float64'].index.values
    ohe_cols = 'popularity'
    complete_feature_set = create_feature_set(spotify_df, float_cols=float_cols)
    print("main")


if __name__ == "__main__":
    main()