#!/usr/bin/env python
"""
Utilities for downloading data from Standforw LMRD, tokenizing, creating vocabulary, encoding and decoding sentences.

modified copy of https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/data_utils.py

__author__ = "Valentin Lievin, DTU, Denmark"
__copyright__ = "Copyright 2017, Valentin Lievin"
__credits__ = ["Valentin Lievin"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Valentin Lievin"
__email__ = "valentin.lievin@gmail.com"
__status__ = "Development"

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import sys
from tqdm import tqdm
import spacy
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

# URLs for WMT data.
_DATA_URL_ = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_DATA_DIR_ = 'data_WMT/'
_RAW_SENTENCES_DIR_ = _DATA_DIR_ + 'raw_sentences/'
_SENTENCES_DIR = _DATA_DIR_ + 'sentences/'
_VOCAB_DIR_ = _DATA_DIR_+'vocab.dat'

nlp = spacy.load('en')
character_pattern = re.compile('([^\s\w\'\.\!\,\?]|_)+')

def to_unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


def cleanHTML(raw_html):
    """
    Remove HTML tags
    """
    return BeautifulSoup(raw_html,"lxml").text

def sentence_tokenizer(text):
    """
    split a text into a list of sentences
    Args:
        text: input text
    Return:
        list of sentences
    """
    return sent_tokenize(text)

def character_tokenizer(sentence):
    """
    character tokenizer
    
    Remove non alphanumeric characters, lowercase and split
    Args:
        sentence: String. input to be processed
    Return:
        a list of characters
    """
    sentence = character_pattern.sub('', sentence)
    sentence = sentence.replace(' ', '_')
    return list(sentence.lower())

def maybe_download(directory, filename, url):
    """Download filename from url unless it's already in directory."""
    if not os.path.exists(directory):
        print("Creating directory %s" % directory)
        os.mkdir(directory)
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        print("Downloading %s to %s" % (url, filepath))
        filepath, _ = urllib.request.urlretrieve(url, filepath)
        statinfo = os.stat(filepath)
        print("Successfully downloaded", filename, statinfo.st_size, "bytes")
    return filepath

def gunzip_file(gz_path, new_path):
    """Unzips from gz_path into new_path."""
    print("Unpacking %s to %s" % (gz_path, new_path))
    with gzip.open(gz_path, "rb") as gz_file:
        with open(new_path, "wb") as new_file:
            for line in gz_file:
                new_file.write(line)
        
def getData(directory):
    """Download the data unless it's already there"""
    train_path = directory
    corpus_file = maybe_download(directory, "training-giga-fren.tar",
                                 _DATA_URL_)
    if not os.path.isdir(_RAW_SENTENCES_DIR_):
        print("Extracting tar file %s" % corpus_file)
        with tarfile.open(corpus_file, "r") as corpus_tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(corpus_tar, directory)
        os.makedirs(_RAW_SENTENCES_DIR_)
        gunzip_file(train_path + "giga-fren.release2.fixed.en.gz", _RAW_SENTENCES_DIR_+'sentences.txt')
            
    else:
        print("Data already downloaded.")

def create_vocabulary(vocabulary_path, data_paths, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    """Create vocabulary file (if it does not exist yet) from data file.
      Data files are supposed to be a list of files with the list of directories. Each sentence is
      tokenized and digits are normalized (if normalize_digits is set).
      Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
      We write it to vocabulary_path in a one-token-per-line format, so that later
      token in the first line gets id=0, second line gets id=1, and so on.
      Args:
        vocabulary_path: path where the vocabulary will be created.
        data_path: data file that will be used to create vocabulary.
        max_vocabulary_size: limit on the size of the created vocabulary.
        tokenizer: a function to use to tokenize each data sentence;
          if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
     """
    if not gfile.Exists(vocabulary_path):
        vocab = {}
        files = []
        files += [data_paths+f for f in os.listdir(data_paths) ]
        for one_file in files:
            with gfile.GFile(one_file, mode="rb") as f:
                review = f.read()
                tokens = tokenizer(review) if tokenizer else character_tokenizer(review)
                for w in tqdm(tokens):
                    word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")
    else:
        print("Vocabulary already created.")
                

def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.
      We assume the vocabulary is stored one-item-per-line, so a file:
        dog
        cat
      will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
      also return the reversed-vocabulary ["dog", "cat"].
      Args:
        vocabulary_path: path to the file containing the vocabulary.
      Returns:
        a pair: the vocabulary (a dictionary mapping string to integers), and
        the reversed vocabulary (a list, which reverses the vocabulary mapping).
      Raises:
        ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.
      For example, a sentence "I have a dog" may become tokenized into
      ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
      "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
      Args:
        sentence: the sentence in bytes format to convert to token-ids.
        vocabulary: a dictionary mapping tokens to integers.
        tokenizer: a function to use to tokenize each sentence;
          if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
      Returns:
        a list of integers, the token-ids for the sentence.
    """
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = character_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    output =  [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]
    output += [EOS_ID]
    return output

def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.
      This function loads data line-by-line from data_path, calls the above
      sentence_to_token_ids, and saves the result to target_path. 
      Sentiment scores are added using the file names ([[id]_[rating].txt])
      See comment for sentence_to_token_ids on the details of token-ids format.
      Args:
        data_path: path to the data file in one-sentence-per-line format.
        target_path: path where the file with token-ids will be created.
        vocabulary_path: path to the vocabulary file.
        tokenizer: a function to use to tokenize each sentence;
          if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if not gfile.Exists(target_path+"sentences.txt"):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        files = []
        #for d in data_path:
        files += [data_path+f for f in os.listdir(data_path) ]
        with gfile.GFile(target_path+"sentences.txt" , mode="w") as tokens_file:
            for one_file in files:
                with gfile.GFile(one_file, mode="rb") as f:
                    for line in tqdm(f.readlines()):
                        sentence = cleanHTML( line )
                        if len(sentence) > 10: # don't save short sentences
                            while sentence[0] == " ":
                                sentence = sentence[1:]
                            token_ids = sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab,
                                                        tokenizer, normalize_digits)
                            tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
                            #sentiments_files.write( str(rating) + "\n")
                            
def prepare_data(vocabulary_size):
    """
    Download the Large Movie Review Dataset, create the vocabulary 
    and convert every sentence in the dataset into list of ids
    
    Args:
        vocabulary_size: maximum number words in the vocabulary
    """
    print("Downloading data from " + _DATA_DIR_ +"..")
    getData(_DATA_DIR_)
    print("Creating Vocabulary..")
    create_vocabulary( _VOCAB_DIR_, _RAW_SENTENCES_DIR_, vocabulary_size )
    print("Converting sentences to sequences of ids..")
    data_to_token_ids( _RAW_SENTENCES_DIR_ , _SENTENCES_DIR, _VOCAB_DIR_ )
    

def read_data(max_size=None, max_sentence_size=None, min_sentence_size=10):
    """Read data from source.
    Args:
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).
    max_sentence_size: maximum size of sentences
    Returns:
    data_set: training data
    """
    sentences = []
    with tf.gfile.GFile('data_WMT/sentences/sentences.txt', mode="r") as source_file:
        source = source_file.readline()
        print (source)
        counter = 0
        while source and (not max_size or counter < max_size):
            source_ids = [int(x) for x in source]
            if len(source_ids) < max_sentence_size and len(source_ids) > min_sentence_size:
                sentences.append(source_ids)
                ratings.append(rating)
                counter += 1
                if counter % 10000 == 0 and counter != 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
            source = source_file.readline()
        return sentences
    
class EncoderDecoder:
    """
    A class to encode text to a sequence of ids or to decode a sequence of ids to text
    """
    def __init__(self):
        """
        Load vocabulary
        """
        self.vocab,self.rev_vocab = initialize_vocabulary(_VOCAB_DIR_)
        
    def encode(self, sentence):
        """
        Encode a sentence to a sequence of ids
        """
        return sentence_to_token_ids(sentence, self.vocab)
    
    def decode(self, seq):
        """
        Decode a sequence of ids to a sentence
        """
        return [ self.rev_vocab[int(el)] for el in seq ]
    
    def prettyDecode(self,seq):
        """
        decode and return a nicely formatted string
        """
        s = "".join(self.decode(seq))
        s = s.replace("_EOS", "" )
        s = s.replace("_PAD", "" )
        s = s.replace("_", " " )
        return s
    
    def vocabularySize(self):
        """
        return the number of unique symbols in the vocabulary (useful for oneHot encoding)
        """
        return len(self.vocab.keys())
