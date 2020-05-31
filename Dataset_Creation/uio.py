import os

path = os.path.expanduser('abc.pdf')
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk import RegexpTokenizer
word_tokenize = RegexpTokenizer("[\w']+")
reader = CategorizedPlaintextCorpusReader(path,r'.*\.txt',cat_pattern=r'(.*)_.*',word_tokenizer=word_tokenize)
print(reader.words(fileids=None,categories=None))
#reader.sents(categories='pos')