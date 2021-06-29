import unittest
from preprocessing import *


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv('../data/results_genre.csv')

    def test_add_chart_info(self):
        df = add_chart_info(self.df, '../data/charts.csv')
        charts = pd.read_csv('../data/charts.csv')
        charts.date = pd.to_datetime(charts.date, format='%Y-%m-%d')
        test_song = "Poor Little Fool"
        test_rank = charts[charts.song == test_song]
        test_rank_sum = test_rank['rank'].sum()
        total_num_weeks = abs((charts.iloc[0].date - charts.iloc[-1].date).days / 7)
        test_rank = (test_rank_sum + (101 * (total_num_weeks - len(test_rank)))) / total_num_weeks
        self.assertEqual(df[df.song == test_song]['rank_alltime'][0], test_rank)

    def test_songs_for_author(self):
        df = clean_dataset(self.df)
        artists = df.drop_duplicates('artist')
        self.assertTrue(all((df.artist == artist).sum() > 10 for artist in artists.artist))

    def test_artist_mean_embedding(self):
        df = clean_dataset(self.df)
        glove = Glove('glove.twitter.27B.')
        artist_mean = get_artist_mean_embeddings(df, glove)
        self.assertIs(type(list(artist_mean.keys())[0]), str)
        self.assertTrue(all(val.shape == (100,) for val in artist_mean.values()))


if __name__ == '__main__':
    unittest.main()
