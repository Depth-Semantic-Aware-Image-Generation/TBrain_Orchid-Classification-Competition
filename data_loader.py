# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:48:12 2022

@author: Peimao
"""

import os
import csv
from itertools import islice
import shutil

path_img = 'C:/flower_category/training'
ls = os.listdir(path_img)
len1 = len(ls)

path = 'training/label.csv'


img_category = []
f = open(path, 'r')
rows = csv.reader(f, delimiter=',')
for category in islice(rows,1,None):
    path = 'C:\\flower_category\\category\\' + category[1]
    if not os.path.isdir(path):
        os.mkdir(path)
    shutil.move(path_img + '/' + category[0], 'C:/flower_category/category' + '/' + category[1] + '/' + category[0])

