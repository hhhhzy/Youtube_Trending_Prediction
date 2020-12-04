# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import text2vec
import preprocessings


# given a label and how many total labels, return the range of views
def get_views(df, label, num_labels):
  a = df['view_count_log'].quantile(label/num_labels)
  b = df['view_count_log'].quantile((label+1)/num_labels)
  range_views = [int(np.exp(a)), int(np.exp(b))]
  return range_views
