from preprocessing import *
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re
import typing
import numpy as np



class Filters:
    filters_pattern = re.compile(r'([\w\.]+):(.+)')
    def __init__(self, filters: typing.List[str]):
        self.filters = []
        for filt in filters:
            match = self.filters_pattern.match(filt)
            self.filters.append(self.Filter(match.group(1), match.group(2)))

    def apply_all(self, df: pd.DataFrame):
        for filt in self.filters:
            df = filt.apply(df)
        return df

    class Filter:
        def __init__(self, field: str, value: str):
            self.field = field
            self.values = value.split(',')

        def apply(self, df: pd.DataFrame):
            mask = pd.Series(False, index=np.arange(len(df)))
            for value in self.values:                    
                if 'top' in value:
                    k = int(value.split(' ')[-1])
                    top_values = df.groupby(self.field).size().sort_values(ascending=False)[:k]
                    mask = mask | (df[self.field].isin(top_values.index))
                else:    
                    mask = mask | (df[self.field] == value)
            return df[mask]


def get_whole_matrix(emb_dict):
    map_idx = {el: i for i, el in enumerate(sorted(emb_dict.keys()))}
    return np.vstack([emb_dict[k] for k in sorted(emb_dict.keys())]), map_idx



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot random embedding-related stuff for authors or songs.\nA few examples for --filters:' \
        '\n-f "artist:The Beatles,The Doors" -- This takes only the beatles and the doors.' \
        'Comma separated filters are joined via OR, space separated are applied in given order.', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', '--type', dest='emb_type', choices=['artists', 'songs'], help='work with artists or songs', required=True)
    parser.add_argument('-f', '--filters', dest='filters', nargs='*', default=tuple(), help='(optionally) specify how to filter data. syntax: "field:value"')
    parser.add_argument('-d', '--dimension', dest='dim', default=100, type=int, help='GloVe embeddings dimension. Default 100')
    parser.add_argument('--height', dest='height', default=13, type=int, help='height for the matplotlib backend.')
    parser.add_argument('--font-size', dest='font_size', default=13, type=int, help='font size for text annotations')
    parser.add_argument('-a', '--aspect', dest='aspect', default=0.8, type=float, help='aspect * height = width')
    parser.add_argument('--title', dest='title', help='Title of the plot', default='')
    parser.add_argument('-o', '--output', dest='output', help='output path of the plot', required=True)
    args = parser.parse_args()
    filters = Filters(args.filters)

    df = fetch_dataset(None, 'data/results_genre.csv', clean_genre=False)
    glove = Glove('glove.twitter.27B.', dim=args.dim)
    df = filters.apply_all(df)
    if args.emb_type == 'songs':
        embedding_mean = get_lyrics_mean_embeddings(df, glove)
    else:
        embedding_mean = get_artist_mean_embeddings(df, glove)
    whole_matrix, _ = get_whole_matrix(embedding_mean)
    tsne = TSNE(n_components=2, random_state=42)
    print('Fitting TSNE')
    low_dim_data = tsne.fit_transform(whole_matrix)

    tsne_df = pd.DataFrame(low_dim_data, sorted(embedding_mean.keys()), columns=['0', '1'])
    km = KMeans(n_clusters=4, random_state=42)
    labels = km.fit_predict(low_dim_data)
    tsne_df['cluster'] = labels

    fg = sns.FacetGrid(data=tsne_df, hue='cluster', height=args.height, aspect=args.aspect)
    fg = fg.map(plt.scatter, '0', '1').add_legend()    

    for i, k in enumerate(sorted(embedding_mean.keys())):
        if len(k) > 30:
            k = k[:25] + "..."
        fg.axes[0][0].annotate(k, (tsne_df.iloc[i, 0], tsne_df.iloc[i, 1]), size=args.font_size)

    plt.title(args.title)
    plt.savefig(args.output)
