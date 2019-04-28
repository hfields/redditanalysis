import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

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

# run this function, input is the name of the csv file
# anacigin is the comments line by line
# anneannen is the controvesiality scores line by line
# Controversiality scores are changed from 0 and 1 to -1 and 1.
# don't forget to change filenames in twitter.py
#f_name is the csv filename
def convert(f_name):
    df = pd.read_csv(f_name, dtype=COLUMN_TO_TYPE, encoding='utf-8')
    all_bodies = df['body'].values

    all_cont = df['controversiality'].values
    with open("anacigin.txt", "w") as anan:
        with open("anneannen.txt", "w") as labels:
            for body, cont in zip(all_bodies, all_cont):           
                try:
                    comment = body.replace("\n"," ")
                    comment = comment.replace("\r"," ")
                    anan.write(comment+"\n")
                    label = cont
                    if label == 0:
                        label = -1
                    labels.write(str(label)+"\n")

                except:
                    continue

#it's like the function above but for pussies
#cutoff is the number of comments you want to keep
def pussyconvert(f_name, cutoff):
    df = pd.read_csv(f_name, dtype=COLUMN_TO_TYPE, encoding='utf-8')
    all_bodies = df['body'].values

    all_cont = df['controversiality'].values
    with open("anacigin.txt", "w") as anan:
        with open("anneannen.txt", "w") as labels:
            i = 0
            for body, cont in zip(all_bodies, all_cont):    
                if i > cutoff:
                    break       
                try:
                    comment = body.replace("\n"," ")
                    comment = comment.replace("\r"," ")
                    anan.write(comment+"\n")
                    label = cont
                    if label == 0:
                        label = -1
                    labels.write(str(label)+"\n")
                    i += 1
                except:
                    continue