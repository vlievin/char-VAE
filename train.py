#!/usr/bin/env python
"""
Train the VRAE model on the large movie review dataset

__author__ = "Valentin Lievin, DTU, Denmark"
__copyright__ = "Copyright 2017, Valentin Lievin"
__credits__ = ["Valentin Lievin"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Valentin Lievin"
__email__ = "valentin.lievin@gmail.com"
__status__ = "Development"
"""    


import tensorflow as tf
import seaborn as sns
import matplotlib
import numpy as np
import time
import datetime 
import json
import os
import data_utils_LMR
from data_utils_LMR import prepare_data,read_data, EncoderDecoder
from model import Vrae as Vrae_model
from training_utilities import BetaGenerator, LearningRateControler
from batch import Generator

# sentiment analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentimentAnalyzer = SentimentIntensityAnalyzer()
def getSentimentScore(sentence):
    scores = sentimentAnalyzer.polarity_scores(sentence)
    return (scores['neg'], scores['neu'] ,scores['pos'])


# flags
tf.app.flags.DEFINE_integer( "char2word_state_size", 512, "char2word hidden state size ")
tf.app.flags.DEFINE_integer( "char2word_num_layers", 2, "char2word num layers ")
tf.app.flags.DEFINE_integer( "encoder_state_size", 1024, "encoder RNN hidden state size")
tf.app.flags.DEFINE_integer( "encoder_num_layers", 1, "encoder RNN num layers ")
tf.app.flags.DEFINE_float("initial_learning_rate", 0.001, "initial learning rate")
tf.app.flags.DEFINE_integer("decoder_state_size", 1024, "state size for the RNN cells (decoder)")
tf.app.flags.DEFINE_integer("decoder_num_layers", 2, "number of layers used in the RNN cells (decoder)")
tf.app.flags.DEFINE_float("learning_rate_change_rate", 3000, "after a changement of hyper-parameters during training, the learning rate stays fixed during this number of steps.")
tf.app.flags.DEFINE_integer("latent_dim", 16, "dimension of the latent space")
tf.app.flags.DEFINE_integer("batch_size", 1048, "length of each batch")
tf.app.flags.DEFINE_integer("sequence_min", 8, "minimum number of characters")
tf.app.flags.DEFINE_integer("sequence_max", 45, "maximum number of characters")
tf.app.flags.DEFINE_integer("epoches", 10000, "Number of epoches")
tf.app.flags.DEFINE_integer("acceptable_accuracy", 0.4, "Increase sentences length when the model reaches this accuracy")
tf.app.flags.DEFINE_integer("input_keep_prob", 0.9, "Dropout keep prob for inputs")
tf.app.flags.DEFINE_integer("output_keep_prob", 0.5, "Dropout keep prob for outpus")
tf.app.flags.DEFINE_string("cell", "LSTM", "cell type: LSTM,GRU,LNLSTM,UGRNN")
tf.app.flags.DEFINE_boolean("peephole",True,"use peephole for LSTM")
tf.app.flags.DEFINE_integer("beta_offset", 15, "number of epoches before increasing Beta.")
tf.app.flags.DEFINE_integer("beta_period", 100, "Beta will be increased from 0 to 1 during this period.")
tf.app.flags.DEFINE_boolean("use_sentiment_feature", True, "Input sentiment features in the stochastic layer.")
tf.app.flags.DEFINE_boolean("teacher_forcing", True, "Teacher forcing increases short term accuracy but penalizes long term gradient probagation.")
tf.app.flags.DEFINE_float("latent_loss_weight", 0.005, "weight used to weaken the latent loss.")
tf.app.flags.DEFINE_integer("dtype_precision", 32, "dtype to be used: typically 32 or 16")
tf.app.flags.DEFINE_boolean("initialize", True, "Initialize model or try to load existing one")
tf.app.flags.DEFINE_string("training_dir" , "sentiment_input_deep", "repertory where checkpoints are logs are saved")
FLAGS = tf.app.flags.FLAGS

if FLAGS.training_dir == "auto":
    FLAGS.training_dir = "logs/state"+str(FLAGS.state_size)+"_layers"+str(FLAGS.num_layers)+"_latent"+str(FLAGS.latent_dim)+"_batch"+str(FLAGS.batch_size)+"_"+str(FLAGS.cell)+"_seqs"+str(FLAGS.sequence_min)+"-"+str(FLAGS.sequence_max)+"_"+str(FLAGS.initial_learning_rate)[-1]+"e"+str(int(np.log10(FLAGS.initial_learning_rate)))+"_B"+str(FLAGS.latent_loss_weight)+"_f"+str(FLAGS.dtype_precision)
else:
    FLAGS.training_dir = "logs/"+FLAGS.training_dir
    
seq_max_init = min( 30, FLAGS.sequence_max)
sequence_max_max = FLAGS.sequence_max
FLAGS.sequence_max = seq_max_init
    
training_parameters = dict()

if FLAGS.initialize:
    #print "||" + str(os.listdir(FLAGS.training_dir))
    print "Checkin in: " + str(FLAGS.training_dir)
    assert not os.path.isdir(FLAGS.training_dir) or len(os.listdir(FLAGS.training_dir)) == 0
    print "logging in: " + str(FLAGS.training_dir)
    training_parameters['step'] = 0
    training_parameters['epoch'] = 0
    training_parameters['n_epoches_since_last_dataset_update'] = 0
    training_parameters['seq_max'] = FLAGS.sequence_max
    training_parameters['learning_rate'] = FLAGS.initial_learning_rate
else:
    print "Checkin in: " + str(FLAGS.training_dir)
    assert os.path.isdir(FLAGS.training_dir) and len(os.listdir(FLAGS.training_dir)) >0
    print "found: " + str(FLAGS.training_dir)
    with open(FLAGS.training_dir +'/training_parameters.json', 'r') as fp:
        training_parameters = json.loads( fp.read() )
    training_parameters['n_epoches_since_last_dataset_update'] = 0
    training_parameters['learning_rate'] = 0.0005 #2e-4#FLAGS.initial_learning_rate
    

# save details
if not os.path.exists(FLAGS.training_dir):
    os.makedirs(FLAGS.training_dir)
flags = dict()
for k,v in FLAGS.__dict__['__flags'].iteritems():
    flags[k] = str(v)
with open(FLAGS.training_dir +'/flags.json', 'w') as fp:
    json.dump( flags , fp)
    
prepare_data(1000)

sentences, ratings = read_data( max_size=None, max_sentence_size=training_parameters['seq_max'],min_sentence_size=FLAGS.sequence_min) 
print len(sentences), " sentences"

# vocabulary encoder-decoder
encoderDecoder = EncoderDecoder()
num_symbols = encoderDecoder.vocabularySize()
# batch generator
space_symbol = encoderDecoder.encode("I am")[1]
word_delimiters = [ data_utils_LMR._EOS, data_utils_LMR._GO, space_symbol ]
batch_gen = Generator(sentences, ratings, FLAGS.batch_size, word_delimiters)
#sentences = [ [1,2,3,0,1,4,5,0] , [1,2,3,0,1,4,5,0] , [1,2,3,0,1,4,5,0] ]
#ratings = [1,2,3]
#batch_gen = Generator(sentences, ratings, 3, 0)
batch_gen.shuffle()
num_iters = FLAGS.epoches * batch_gen.iterations_per_epoch()
# deterministic warm-up control
beta_T = FLAGS.beta_period*batch_gen.iterations_per_epoch()
beta_u = FLAGS.beta_offset*batch_gen.iterations_per_epoch()
betaGenerator = BetaGenerator(num_iters, beta_T, beta_u)
# text decoder ( text <-> ids)
encoderDecoder = EncoderDecoder()
# learning rate
learningRateControler = LearningRateControler(training_parameters['learning_rate'], FLAGS.learning_rate_change_rate, 0.5)

# load model
vrae_model = Vrae_model(char2word_state_size = FLAGS.char2word_state_size, 
                     char2word_num_layers = FLAGS.char2word_num_layers, 
                     encoder_state_size = FLAGS.encoder_state_size, 
                     encoder_num_layers = FLAGS.encoder_num_layers, 
                     decoder_state_size = FLAGS.decoder_state_size, 
                     decoder_num_layers = FLAGS.decoder_num_layers, 
                     latent_dim = FLAGS.latent_dim, 
                     batch_size = FLAGS.batch_size, 
                     num_symbols=num_symbols, 
                     input_keep_prob = FLAGS.input_keep_prob,
                     output_keep_prob = FLAGS.output_keep_prob, 
                     latent_loss_weight = FLAGS.latent_loss_weight, 
                     dtype_precision = FLAGS.dtype_precision, 
                     cell_type = FLAGS.cell, 
                     peephole = False, 
                     sentiment_feature = FLAGS.use_sentiment_feature,
                     teacher_forcing=True)

config = tf.ConfigProto(
        #device_count = {'GPU': 0},
        log_device_placement = False
    )

#training_parameters['seq_size'] = 


# log time
start = time.time()
saver = tf.train.Saver()
init_op = tf.global_variables_initializer()
checkpoint_path = FLAGS.training_dir + '/model.ckp'
try:
    with tf.Session(config=config) as sess:
        # summary writer
        summary_writer = tf.summary.FileWriter(FLAGS.training_dir, sess.graph)
        # init
        tf.set_random_seed(42)
        if FLAGS.initialize:
            sess.run(init_op)
        else:
            saver.restore(sess, "./"+FLAGS.training_dir+'/model.ckp')
        while training_parameters['epoch'] < FLAGS.epoches:
            batch_gen.shuffle()
            while not batch_gen.epochCompleted():
                # get batch
                padded_batch_xs, batch_ys, batch_lengths, batch_weights, end_of_words, batch_word_lengths, max_length = batch_gen.next_batch()
                # sentiment batch
                vaderSentiments = [ getSentimentScore(encoderDecoder.prettyDecode(xx)) for xx in padded_batch_xs]
                training_parameters['learning_rate'] = learningRateControler.learning_rate
                beta = 0.001 + betaGenerator(training_parameters['step']) # add small value to avoid points to scatter
                _,d,loss_reconstruction, loss_regularization, summary,_ = vrae_model.step(sess, 
                                                                                          padded_batch_xs, 
                                                                                          beta, 
                                                                                          training_parameters['learning_rate'], 
                                                                                          batch_lengths, 
                                                                                          batch_weights, 
                                                                                          end_of_words,
                                                                                          batch_word_lengths,
                                                                                          training_parameters['epoch'],
                                                                                         vaderSentiments)
                if training_parameters['step'] > beta_T+beta_u: 
                    learningRateControler.update(d)
                summary_writer.add_summary(summary, global_step=training_parameters['step'])
                if training_parameters['step'] % 10 == 0:
                    print("loss: " + str(d) + " | step: " + str(training_parameters['step'])  + " | beta: " + str(beta) + " | learning rate: " + str(learningRateControler.learning_rate) )
                training_parameters['step'] += 1 
                
                # increase sentences size
                if training_parameters['n_epoches_since_last_dataset_update'] > 10 and loss_reconstruction < FLAGS.acceptable_accuracy and training_parameters['seq_max'] < sequence_max_max:
                    print "###########################\nUPDATING SENTENCES SIZE"
                    FLAGS.training_dir + '/model'+str(training_parameters['seq_max'])+'.ckp'
                    training_parameters['n_epoches_since_last_dataset_update'] = 0
                    training_parameters['seq_max'] += 1
                    sentences, ratings = read_data( max_size=None,max_sentence_size=training_parameters['seq_max'],
                                                   min_sentence_size=FLAGS.sequence_min) 
                    batch_gen = Generator(sentences, ratings, FLAGS.batch_size,word_delimiters)
                    learningRateControler.reset()
                
            training_parameters['epoch'] += 1
            checkpoint_path = saver.save(sess, checkpoint_path)
            print "saved to", checkpoint_path
            training_parameters['n_epoches_since_last_dataset_update'] += 1
            with open(FLAGS.training_dir +'/training_parameters.json', 'w') as fp:
                json.dump( training_parameters , fp)
            
        
        
except KeyboardInterrupt:
    print('training interrupted')
end = time.time()
print (str(datetime.timedelta(seconds=end - start)))
                