import csv
import json
import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join, splitext

COLUMN_TO_TYPE = {'created_utc':np.int64,
                  'ups': np.int64,
                  'subreddit_id': str,
                  'link_id': str,
                  'name': str,
                  'score_hidden': bool,
                  'author_flair_css_class': str,
                  'author_flair_text': str,
                  'subreddit': str,
                  'id': str,
                  'removal_reason': str,
                  'gilded': np.int64,
                  'downs': np.int64,
                  'archived': bool,
                  'author': str,
                  'score': np.int64,
                  'retrieved_on': np.int64,
                  'body': str,
                  'distinguished': str,
                  'edited': str,
                  'controversiality': np.int8,
                  'parent_id': str,
                  }

def json_to_csv(f_name):
    data = []
    with open(f_name) as f:
        for line in f:
            data.append(json.loads(line))

    new_csv = '%s.csv' % (splitext(f_name)[0])
    with open(new_csv, "w", newline="", encoding='utf8') as f:
         title = data[0].keys()
         cw = csv.DictWriter(f, title, delimiter=',')
         cw.writeheader()
         cw.writerows(data)

def json_dir_to_csv(dir):
    files = [f for f in listdir(dir) if splitext(f)[1] == ".json"]
    for file in files:
        json_to_csv(join(dir, file))

def subreddit_csv(subreddit, dir):
    subreddit_df = pd.DataFrame()
    files = [join(dir, f) for f in listdir(dir) if splitext(f)[1] == ".csv"]
    for file in files:
        df = pd.read_csv(file, dtype=COLUMN_TO_TYPE)
        df = df.loc[df['subreddit']==subreddit]
        if subreddit_df.empty:
            subreddit_df.rename(columns=df.columns)
        subreddit_df = subreddit_df.append(df, sort=True)
    new_csv = "%s.csv" % (subreddit)
    subreddit_df.to_csv(path_or_buf=new_csv, sep=',', encoding='utf-8')

def word_counts(f_name):
    stop_words=set(stopwords.words("english"))

    df = pd.read_csv(f_name, dtype=COLUMN_TO_TYPE, encoding='utf-8')
    all_bodies = df['body'].values
    word_counts = {}
    for body in all_bodies:
        try:
            words = set(word_tokenize(body, language='english'))
        except:
            print(body)
        filt_words = [w for w in words if w.lower() not in stop_words]
        for word in filt_words:
            if (word[0] not in string.punctuation) and (word != 'lt') and (word != 'gt'):
                if word not in word_counts:
                    word_counts[word] = 1
                else:
                    word_counts[word] += 1
    return word_counts

def word_hist(word_counts, subreddit):
    word_counts = word_counts.items()
    word_counts = sorted(word_counts, key=lambda x: x[1], reverse=True)

    words = [pair[0] for pair in word_counts[:40]]
    frequencies = [pair[1] for pair in word_counts[:40]]
    freq_series = pd.Series.from_array(frequencies)

    # Plot the figure.
    plt.figure(figsize=(12, 8))
    ax = freq_series.plot(kind='bar')
    ax.set_title('Most common words in r/%s' % (subreddit))
    ax.set_xlabel('Word')
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(words)

    def add_value_labels(ax, spacing=5):
        """Add labels to the end of each bar in a bar chart.

        Arguments:
            ax (matplotlib.axes.Axes): The matplotlib object containing the axes
                of the plot to annotate.
            spacing (int): The distance between the labels and the bars.
        """

        # For each bar: Place a label
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            # Number of points between bar and label. Change to your liking.
            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with one decimal place
            label = "{:.1f}".format(y_value)

            # Create annotation
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va)                      # Vertically align label differently for
                                            # positive and negative values.
    add_value_labels(ax)
    plt.show()

def main():
    # json_dir_to_csv('./reddit_data/2015/RC_2015-05/')
    # subreddit_csv('politics', './reddit_data/2015/RC_2015-05/')
    data = word_counts('./politics.csv')
    word_hist(data, 'politics')

if __name__ == "__main__" :
    main()
