#! /usr/bin/env python
# coding : utf-8
from collections import defaultdict
from pprint import pprint

from gensim import corpora

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

# exclude the stop words
stop_list = set('for a of the and to in'.split())
pprint(stop_list)
texts = [[word for word in document.lower().split() if word not in stop_list] for document in documents]

# exclude the words that only appear once

frequency = defaultdict(int)

for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

pprint(texts)

dictionary = corpora.Dictionary(texts)
dictionary.save('deerwester.txt')

print(dictionary)

print(dictionary.token2id)

"""
为了真正将记号化的文档转换为向量，需要：
函数doc2bow()简单地对每个不同单词的出现次数进行了计数，并将单词转换为其编号，然后以稀疏向量的形式返回结果。
因此，稀疏向量[(0, 1), (1, 1)]表示：在“Human computer interaction”
中“computer”(id 0) 和“human”(id 1)各出现一次；其他10个dictionary中的单词没有出现过（隐含的）。

"""
# test purpose
new_doc = 'Human computer interaction'
new_vec = dictionary.doc2bow(new_doc.lower().split())
# print(new_vec)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('deerwester.mm', corpus)

print(corpus)

