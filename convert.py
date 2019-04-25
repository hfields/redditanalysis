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

def convert(filename):

    csv.field_size_limit(sys.maxsize)
    with open("anacigin", "w") as body:
        with open("anneannen", "w") as labels:
            with open(filename, "r") as input:
                i = 0
                for line in  csv.reader(input, quotechar='"', delimiter=',',quoting=csv.QUOTE_ALL, escapechar='\\'):
                    if i < 2510:
                        if i > 2500:
                            print(line)
                    if i == 0:
                        i = 1
                        continue
                    if len(line) < 7:
                        continue
                    comment = line[5]
                    comment = comment.replace('\n',' ')
                    body.write(comment+"\n")
                    label = line[6]
                    if label == 0:
                        label = -1
                    labels.write(label+"\n")
                    i+=1
                #[ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(input)]

# run this function, input is the name of the csv file
# anacigin is the comments line by line
# anneannen is the controvesiality scores line by line
# Controversiality scores are changed from 0 and 1 to -1 and 1.
# don't forget to change filenames in twitter.py
def lol(f_name):
    df = pd.read_csv(f_name, dtype=COLUMN_TO_TYPE, encoding='utf-8')
    all_bodies = df['body'].values
    all_cont = df['controversiality'].values
    with open("anacigin", "w") as anan:
        with open("anneannen", "w") as labels:
            i = 0
            a = False
            for body in all_bodies:
                if not isinstance(body,str):
                    a = True
                    continue
                comment = body.replace("\n"," ")
                comment = comment.replace("\r"," ")
                anan.write(comment+"\n")
                i+=1
            
            for cont in all_cont:
                if a:
                    a = False
                    continue
                label = cont
                if label == 0:
                    label = -1
                labels.write(str(label)+"\n")
        