# this script is use to compute latent representations of the sentences taken from the dataset
# this is required by the Pessimistic Machine

from tensorflow.python.platform import gfile
import numpy as np
import time
import datetime 
import json
from tqdm import tqdm
import os
import tensorflow as tf
import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import data_utils_LMR
from data_utils_LMR import prepare_data,read_data, EncoderDecoder
from model import Vrae as Vrae_model
from batch import Generator

prepare_data(1000)

training_dir = 'logs/'
training_dir += 'no_char2word'

print "using directory: ",
print training_dir

# sentiment analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentimentAnalyzer = SentimentIntensityAnalyzer()
def getSentimentScore(sentence):
    scores = sentimentAnalyzer.polarity_scores(sentence)
    return (scores['neg'], scores['neu'] ,scores['pos'])

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def string2bool(st):
    if st.lower() == "true":
        return True
    else:
        return False
    
    
with open(training_dir +'/flags.json', 'r') as fp:
    FLAGS = dotdict(json.loads( fp.read() ) )
    
for k,v in FLAGS.iteritems():
    print k,':',v
    
with open(training_dir +'/training_parameters.json', 'r') as fp:
    training_parameters = json.loads( fp.read() )
# vocabulary encoder-decoder
encoderDecoder = EncoderDecoder()
num_symbols = encoderDecoder.vocabularySize()
# prepare data
sentences, ratings = read_data( max_size=None, 
                               max_sentence_size=training_parameters['seq_max'],
                               min_sentence_size=int(FLAGS.sequence_min), 
                               test=False) 
print len(sentences), " sentences"

space_symbol = encoderDecoder.encode("I am")[1]
word_delimiters = [ data_utils_LMR._EOS, data_utils_LMR._GO, space_symbol ]
encoderDecoder = EncoderDecoder()

config = tf.ConfigProto(
        device_count = {'GPU': 1}, 
    )

# load model
vrae_model = Vrae_model(char2word_state_size = int(FLAGS.char2word_state_size), 
                     char2word_num_layers = int(FLAGS.char2word_num_layers), 
                     encoder_state_size = int(FLAGS.encoder_state_size), 
                     encoder_num_layers = int(FLAGS.encoder_num_layers), 
                     decoder_state_size = int(FLAGS.decoder_state_size), 
                     decoder_num_layers = int(FLAGS.decoder_num_layers), 
                          latent_dim=int(FLAGS.latent_dim),
                         batch_size=1,
                         num_symbols=num_symbols,
                        latent_loss_weight=float(FLAGS.latent_loss_weight),
                         dtype_precision=FLAGS.dtype_precision,
                        cell_type=FLAGS.cell, 
                        peephole=FLAGS.peephole,
                        input_keep_prob=float(FLAGS.input_keep_prob),
                        output_keep_prob=float(FLAGS.output_keep_prob),
                      sentiment_feature = string2bool(FLAGS.use_sentiment_feature),
                      use_char2word = string2bool(FLAGS.use_char2word) 
                       )

def zToXdecoded(session,z_sample,s_length):
    x_reconstruct = vrae_model.zToX(session,z_sample,s_length)
    return encoderDecoder.prettyDecode( np.argmax(x_reconstruct[0], axis= 1) ) 
saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    saver.restore(sess, "./"+training_dir+'/model.ckp')
    with gfile.GFile(training_dir + "/latent_representations.txt" , mode="w") as latent_representations:
        for sent in tqdm(sentences):
            if len(sent) > 5:
                s = str(encoderDecoder.prettyDecode(sent))
                z = vrae_model.XToz(sess, *encoderDecoder.encodeForTraining(s), sentiment=getSentimentScore(s))[0]
                s += "|"
                for z_ in list(z):
                    s+= str(z_)
                    s+= ","
                s = s[:-1]
                s += "\n"
                latent_representations.write( s)