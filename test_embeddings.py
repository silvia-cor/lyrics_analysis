import unittest

from numpy import single
from embeddings import Filters
from preprocessing import fetch_dataset


class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        self.df = fetch_dataset(None, 'results_genre.csv')

    def test_filters(self):
        single_filter = "artist:The Beatles"
        filters = Filters([single_filter])
        df = filters.apply_all(self.df)
        self.assertTrue((df.artist == 'The Beatles').all())

        single_filter = "artist:The Beatles,The Doors"
        filters = Filters([single_filter])
        df = filters.apply_all(self.df)
        self.assertTrue(((df.artist == 'The Beatles') | (df.artist == 'The Doors')).all())


if __name__ == '__main__':
    unittest.main()
