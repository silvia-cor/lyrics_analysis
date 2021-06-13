from crawler.access_tokens import credentials
from tqdm import tqdm
import lyricsgenius
import time
import pandas as pd
import logging


logging.basicConfig(filename='crawling.log', format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)


if __name__ == '__main__':
    df = pd.read_csv('charts.csv')
    df['lyrics'] = None
    df.drop_duplicates('song', inplace=True)
    genius = lyricsgenius.Genius(credentials['token'])
    genius.verbose = False
    for i, row in tqdm(df.iterrows()):
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


    df.to_csv('results.csv')