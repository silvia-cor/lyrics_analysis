import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.preprocessing_classification import genres_acc


def bin_artists_graphic():
    file_path = f'output/bin_artists_all.csv'
    df = pd.read_csv(file_path)
    new_df = df.nsmallest(5, 'f1')
    df = df.drop(df.nsmallest(5, 'f1').index)
    new_df = new_df.append(df.nlargest(5, 'f1'))
    df = df.drop(df.nlargest(5, 'f1').index)
    new_df = new_df.append(df.sample(15, random_state=15))
    new_df = new_df.sort_values('f1', ascending=False)
    artists = new_df['artists']
    f1 = new_df['f1']
    ticks = [int(i) for i in range(len(artists))]
    fig = plt.figure()
    plt.barh(ticks, f1)
    plt.yticks(fontsize=5.5)
    plt.xlim(0.2)
    plt.margins(y=0.01)
    plt.yticks(ticks, artists)
    plt.yticks(wrap=True)
    plt.title('Results of binary classification on artists')
    plt.xlabel('F1')
    plt.show()


def bin_genres_graphic():
    file_path = f'output/bin_genres_all.csv'
    df = pd.read_csv(file_path)
    df = df.sort_values('f1', ascending=False)
    genres = [genres_acc[i] for i in df['genres']]
    f1 = df['f1']
    ticks = [int(i) for i in range(len(genres))]
    fig = plt.figure()
    plt.barh(ticks, f1)
    plt.yticks(fontsize=7)
    plt.margins(y=0.01)
    plt.xlim(0.4)
    plt.yticks(ticks, genres)
    plt.yticks(wrap=True)
    plt.title('Results of binary classification on genres')
    plt.xlabel('F1')
    plt.show()

