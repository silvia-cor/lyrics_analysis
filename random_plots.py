from preprocessing import *
from tqdm import tqdm
import argparse 
import matplotlib.pyplot as plt
import bar_chart_race as bcr


def from_millis_to_minutes(millis: float) -> float:
    return round(millis / (1000*60), 2)


def from_minutes_to_millis(minutes: float) -> float:
    return (minutes * (1000*60))


def gather_durations_data(df, args):
    data = {}
    if args.decades:
        decades = df.groupby(pd.Grouper(key='date', freq='10YS'))
        for dt, decade in tqdm(decades):
            data[dt.year] = {}
            for duration in durations_millis:
                data[dt.year][f'{from_millis_to_minutes(duration)}'] = (decade.duration > duration).sum()
    else:
        for duration in durations_millis:
            data[from_millis_to_minutes(duration)] = (df.duration > duration).sum()
    return data


def gather_genres_data(df, args):
    df = df[df.genre.notna()]
    df['genre'] = df['genre'].apply(get_genre)
    df = df[df.genre != 'wtf']
    df.genre = df.genre.map(lambda i: genres_acc[i])
    data = {}
    selected_genres = set(args.genres)
    if args.decades:
        decades = df.groupby(pd.Grouper(key='date', freq='10YS'))
        for dt, decade in tqdm(decades):
            data[dt.year] = {}
            if len(args.genres) == 1 and "top" in args.genres[0]:
                top_k = int(args.genres[0].split(' ')[1])
                for genre_name, count in decade.groupby('genre').genre.count().nlargest(top_k).iteritems():
                    genre_name = 'prog. rock' if genre_name == 'progressive rock' else genre_name
                    data[dt.year][genre_name.capitalize()] = count
            else:
                for genre_name, genre in decade.groupby('genre'):
                    if genre_name in selected_genres or 'all' in selected_genres:
                        genre_name = 'prog. rock' if genre_name == 'progressive rock' else genre_name
                        data[dt.year][genre_name.capitalize()] = genre.genre.count()
    else:
        if len(args.genres) == 1 and "top" in args.genres[0]:
            top_k = int(args.genres[0].split(' ')[1])
            for genre_name, count in df.groupby('genre').genre.count().nlargest(top_k).iteritems():
                genre_name = 'prog. rock' if genre_name == 'progressive rock' else genre_name
                data[genre_name.capitalize()] = count
        else:
            for genre_name, genre in df.groupby('genre'):
                if genre_name in selected_genres or 'all' in selected_genres:
                    genre_name = 'prog. rock' if genre_name == 'progressive rock' else genre_name
                    data[genre_name.capitalize()] = genre.genre.count()
    return data


def chart_race_and_exit(args):
    df = pd.read_csv('results_genre.csv')
    df = df[df['artist'] != 'Glee Cast']
    df = df[df.genre.notna()]
    df['genre'] = df['genre'].apply(get_genre)
    df = df[df.genre != 'wtf']
    genres_acc_cap = list(map(lambda g: g.capitalize(), genres_acc))
    df.genre = df.genre.map(lambda i: genres_acc_cap[i])
    df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
    df = df[df.date < datetime(year=2015, month=1, day=1)]
    groups = df.groupby(pd.Grouper(key='date', freq='1YS'))
    data = {}
    for date, group in groups:
        counts = group.genre.value_counts()
        counts = counts.to_dict()
        for g in (set(genres_acc_cap) - set(counts.keys())):
            counts[g] = 0.
        
        data[date] = counts 
    data = pd.DataFrame(data).T
    print("Creating chart race and quitting")
    bcr.bar_chart_race(
        df=data,
        n_bars=8,
        filename='./chart_race_genres.gif',
        sort='desc',
        fixed_order=False,
        steps_per_period=20,
        period_length=800,
        period_label={'x': .98, 'y': .3, 'ha': 'right', 'va': 'center'},
        perpendicular_bar_func=None,
        period_fmt="%Y",
        cmap='dark12',
        bar_size=.70,
        scale='log',
        title=args.title
    )
    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot random stuff duration-related')
    parser.add_argument('-d', '--durations', dest='durations', nargs='+', help='compute number of songs greater than these durations. Format: minutes.seconds')
    parser.add_argument('-g', '--genres', dest='genres', nargs='+', help='plot frequency of specified genres. Use "all" for all genres, use "top k" to select top k genres only')
    parser.add_argument('--chart_race', '-r', dest='chart_race', action='store_true', help='specify to output a bar chart race (only valid when -g is specified)')
    parser.add_argument('--decades', dest='decades', action='store_true', help='specify to group by decades')
    parser.add_argument('--normalize', dest='normalize', action='store_true', help='specify to normalize data')
    parser.add_argument('--decades_on_x', dest='decades_on_x', action='store_true', help='plots decades on x (plots durations on x otherwise)')
    parser.add_argument('--title', dest='title', help='title of the plot figure', default='')
    parser.add_argument('--x_label', dest='x_label', help='x label name', default='')
    parser.add_argument('--y_label', dest='y_label', help='y label name', default='')
    parser.add_argument('--figsize', dest='figsize', help='size of plot figure, eg. 12,10', default='12,10')
    parser.add_argument('--output_path', dest='output_path', help='output path to save plot, eg. decades.png')
    args = parser.parse_args()
    if args.chart_race and not args.genres:
        raise UserWarning("--chart_race specified without --genres. This will be ignored")
    elif args.chart_race and args.genres:
        chart_race_and_exit(args)
    durations_millis = list(map(lambda d: from_minutes_to_millis(float(d)), args.durations))
    df = pd.read_csv('results_genre.csv')
    df = df[df['artist'] != 'Glee Cast']
    df = aggregate_sixties_and_2020s(df)
    _, ax = plt.subplots(figsize=tuple(map(int, args.figsize.split(','))))

    if args.genres:
        data = gather_genres_data(df, args)
    else:
        data = gather_durations_data(df, args)

    data = pd.DataFrame(data) if args.decades else pd.DataFrame(data, index=['Frequency'])
    if args.normalize:
        for ind in data.index:
            data.loc[ind] = data.loc[ind] / data.loc[ind].max()
    if args.decades_on_x:
        data.T.plot.bar(title=args.title, xlabel=args.x_label, ylabel=args.y_label, ax=ax, legend=args.decades)
    else:
        data.plot.bar(title=args.title, xlabel=args.x_label, ylabel=args.y_label, ax=ax, legend=args.decades)

    if args.output_path:
        plt.savefig(args.output_path)
    else:
        plt.show()
