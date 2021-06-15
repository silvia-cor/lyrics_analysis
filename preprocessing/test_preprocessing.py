import unittest
from preprocessing import *


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv('results_genre.csv')

    def test_add_chart_info(self):
        df = add_chart_info(self.df, 'charts.csv')
        charts = pd.read_csv('charts.csv')
        charts.date = pd.to_datetime(charts.date, format='%Y-%m-%d')
        test_song = "Poor Little Fool"
        test_rank = charts[charts.song == test_song]
        test_rank_sum = test_rank['rank'].sum()
        total_num_weeks = abs((charts.iloc[0].date - charts.iloc[-1].date).days / 7)
        test_rank = (test_rank_sum + (101 * (total_num_weeks - len(test_rank)))) / total_num_weeks
        self.assertEqual(df[df.song == test_song]['rank_alltime'][0], test_rank)


if __name__ == '__main__':
    unittest.main()