# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import codecs
import re

import text2vec  # this is a self-written package


def load_data(dataset_path):
  df = pd.read_csv(dataset_path)
  return df


# generating a new column recording the days between publishing and scarping
def get_time_gap(df, trend):
  # first change the format of 'trending_date' to match with 'publishedAt'
  if not trend:
    a = df['trending_date']
    arr = []
    for i in list(a):
      t = i.split('.')
      year = t[0]+'20'
      month = t[2]
      day = t[1]
      s = [year] + [month] + [day]
      s = "-".join(s)
      arr.append(s)
    df['trending_date'] = pd.Series(arr)
    
  df['publishedAt'] = pd.to_datetime(df['publishedAt']).dt.date
  df['trending_date'] = pd.to_datetime(df['trending_date']).dt.date
  df['time_gap'] = df['trending_date'] - df['publishedAt'] 
  # 1 means 0-1 days, 2 means 1-2 days, etc
  df['time_gap'] = df['time_gap'].dt.days + 1
  df['time_gap'].astype(int)
  return df


# Create a new variable of category name mapping from category id
def get_category_name(df):
  df['category_name'] = np.nan
  df.loc[(df['categoryId'] == 1),'category_name'] = 'Film and Animation'
  df.loc[(df['categoryId'] == 2),'category_name'] = 'Cars and Vehicles'
  df.loc[(df['categoryId'] == 10),'category_name'] = 'Music'
  df.loc[(df['categoryId'] == 15),'category_name'] = 'Pets and Animals'
  df.loc[(df['categoryId'] == 17),'category_name'] = 'Sport'
  df.loc[(df['categoryId'] == 19),'category_name'] = 'Travel and Events'
  df.loc[(df['categoryId'] == 20),'category_name'] = 'Gaming'
  df.loc[(df['categoryId'] == 22),'category_name'] = 'People and Blogs'
  df.loc[(df['categoryId'] == 23),'category_name'] = 'Comedy'
  df.loc[(df['categoryId'] == 24),'category_name'] = 'Entertainment'
  df.loc[(df['categoryId'] == 25),'category_name'] = 'News and Politics'
  df.loc[(df['categoryId'] == 26),'category_name'] = 'How to and Style'
  df.loc[(df['categoryId'] == 27),'category_name'] = 'Education'
  df.loc[(df['categoryId'] == 28),'category_name'] = 'Science and Technology'
  df.loc[(df['categoryId'] == 29),'category_name'] = 'Non Profits and Activism'
  df.loc[(df['categoryId'] == 25),'category_name'] = 'News & Politics'
  return df


# normailze ‘view_count’, ‘likes’ and ‘comment_count’
def normalize(df):
  df['likes_log'] = np.log(df['likes']+0.0001)
  df['view_count_log'] = np.log(df['view_count']+0.0001)
  df['dislikes_log'] = np.log(df['dislikes']+0.0001)
  df['comment_log'] = np.log(df['comment_count']+0.0001) 
  return df


# Convert True/False columns (e.g. ‘comments_disabled’) into binary variables
def to_bi(df):
  df['comments_disabled'] = df['comments_disabled'].astype(int)
  df['ratings_disabled'] = df['ratings_disabled'].astype(int)
  return df


# embed text variables('title','tags','description') with pretrained glove
def get_embedded_text(df, pretrained_glove_path):
  # given a pretrained glove model, returns dataframe with new columns which are vectors embedded from all text variables
  glove_model = text2vec.loadGloveModel(pretrained_glove_path)
  title = df['title'].astype(str)
  tags = df['tags'].astype(str)
  description = df['description'].astype(str)
    
  df['title_embedded'] = text2vec.embedding(title, glove_model).tolist()
  df['tags_embedded'] = text2vec.embedding(tags, glove_model).tolist()
  df['description_embedded'] = text2vec.embedding(description, glove_model).tolist()
  return df


# drop useless columns
def drop_column(df):
  df = df.drop(['thumbnail_link',  'channelId', 'view_count', 'likes',
           'dislikes', 'comment_count', 'publishedAt', 'trending_date', 'channelTitle'], axis=1)
  return df


# generate labels
def generate_label(df, num_labels):
  df['label'] = 0
  for i in range(num_labels):
    a = df['view_count_log'].quantile(i/num_labels)
    b = df['view_count_log'].quantile((i+1)/num_labels)
    df.loc[(df['view_count_log'] < b) & (df['view_count_log'] >= a), 'label'] = i
  return df

# generate a preprocessed dataframe to fit in predictive models
def get_preprocessed_data(dataset_path, pretrained_glove_path, num_labels, trend = False):
  df = load_data(dataset_path)
  df = get_category_name(df)
  df = get_time_gap(df, trend)
  df = df.drop_duplicates(subset=['video_id'])
  df = to_bi(df)
  df = normalize(df)
  df = get_embedded_text(df, pretrained_glove_path)
  df = generate_label(df, num_labels)
  df = drop_column(df)

  return df
