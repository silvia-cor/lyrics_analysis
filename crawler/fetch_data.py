from crawler.access_tokens import credentials
from tqdm import tqdm
from typing import Optional, List, Dict
import pylast
import lyricsgenius
import time
import pandas as pd
import logging
import requests
import time

logging.basicConfig(filename='crawling.log', format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)


def __get_track_info(mbid: str, lastfm_net: pylast.LastFMNetwork) -> Optional[Dict]:
    url = f'https://{"".join(lastfm_net.ws_server)}'
    req = requests.get(url, params={'method': 'track.getinfo', 'api_key': lastfm_net.api_key, 'mbid': mbid, 'format': 'json'})
    if req.status_code == 200:
        json = req.json()
        genres = list(map(lambda t: t['name'], json['track']['toptags']['tag']))
        return {'genres': genres, 'duration': json['track']['duration']}
    return None


def fetch_lyrics(csv_path='charts.csv', output_path='results.csv'):
    df = pd.read_csv(csv_path)
    df['lyrics'] = None
    df.drop_duplicates('song', inplace=True)
    genius = lyricsgenius.Genius(credentials['genius']['token'])
    genius.verbose = False
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            song = genius.search_song(title=row.song.lower(), artist=row.artist.lower())
        except Exception as e:
            logging.error(e)
            continue
        if song is None:
            logging.error(f'Could not retrieve lyrics for {row.song}, by {row.artist}')
            continue
        df.loc[i, 'lyrics'] = song.lyrics
        time.sleep(0.5)


    df.to_csv(output_path)


def fetch_genre(csv_path='results.csv', output_path='results_genre.csv'):
    df = pd.read_csv(csv_path)
    df['genres'] = None
    df['duration'] = None
    lastfm = pylast.LastFMNetwork(api_key=credentials['lastfm']['key'], api_secret=credentials['lastfm']['secret'])
    for i, row in tqdm(df.iterrows(), total=len(df)):
        time.sleep(0.5)
        try:
            track = lastfm.search_for_track(track_name=row.song.lower(), artist_name=row.artist.lower()).get_next_page()[0]
            if track is None:
                logging.error(f'Could not retrieve last.fm track for {row.song}, by {row.artist}')
                continue
            info = __get_track_info(track.get_mbid(), lastfm)
            if info is not None:
                df.loc[i, 'genre'] = ':'.join(info['genres'])
                df.loc[i, 'duration'] = info['duration']
            else:
                logging.error(f'Could not retrieve last.fm info for {row.song}, by {row.artist}')
            
        except Exception as e:
            logging.error(e)
            continue
    df.to_csv(output_path)


if __name__ == '__main__':
    fetch_genre()
