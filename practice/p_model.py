#! /usr/bin/env python
# coding : utf-8

import csv
import random
import re
import string
from collections import namedtuple

import numpy as np
from gensim.models import doc2vec

# read wikipedia csv data
reader = csv.reader(open("wikipedia.csv"))
count = 0
data = ''

for row in reader:
    count = count + 1
    if count > 301:
        break
    else:
        data += row[1]

# split the sentence, end with " . ? ! ".
sentence_enders = re.compile('[.?!]')
data_list = sentence_enders.split(data)

label_doc = namedtuple('label_doc', 'words tags')
exclude = set(string.punctuation)

all_docs = []
count = 0

for sen in data_list:
    word_list = sen.split()
    # will ignore the sentence if the words of this sentence less than 3 words
    if len(word_list) < 3:
        continue

    tag = ['SEN_' + str(count)]
    count += 1
    sen = ''.join(ch for ch in sen if ch not in exclude)
    all_docs.append(label_doc(words=sen.split(), tags=tag))

print(all_docs)

# use fixed learning rate
model = doc2vec.Doc2Vec(alpha=0.025, min_alpha=0.025)
model.build_vocab(all_docs)
for epoch in range(10):
    model.train(all_docs)
    # decrease then learning rate
    model -= 0.002
    model.min_alpha = model.alpha

model.save('my_model.doc2vec')


def train_model():
    doc_id = random.randint(model.docvecs.count)
    print(doc_id)

    sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)
    print('TARGET', all_docs[doc_id].words)
    count = 0

    for i in sims:
        if count > 8:
            break
        pid = int(string.replace(i[0], "SEN_", ""))
        print(i[0], ": ", all_docs[pid].words)
        count += 1