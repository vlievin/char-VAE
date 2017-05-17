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
from data_utils_LMR import prepare_data,read_data, EncoderDecoder
from model import Vrae as Vrae_model
from training_utilities import BetaGenerator, LearningRateControler
from batch import Generator

# flags
tf.app.flags.DEFINE_float("initial_learning_rate", 0.0008, "initial learning rate")
tf.app.flags.DEFINE_float("learning_rate_change_rate", 1500, "learning rate can be updated after this number of iterations")
tf.app.flags.DEFINE_integer("state_size", 1024, "state size for the RNN cells (used both for encoder and decoder)")
tf.app.flags.DEFINE_integer("num_layers", 2, "number of layers used in the RNN cells (used both for encoder and decoder)")
tf.app.flags.DEFINE_integer("latent_dim", 16, "dimension of the latent space")
tf.app.flags.DEFINE_integer("batch_size", 512, "length of each batch")
tf.app.flags.DEFINE_integer("sequence_min", 15, "minimum number of characters")
tf.app.flags.DEFINE_integer("sequence_max", 30, "maximum number of characters")
tf.app.flags.DEFINE_integer("epoches", 10000, "Number of epoches")
tf.app.flags.DEFINE_integer("acceptable_accuracy", 0.8, "Increase sentences length when the model reach this accuracy")
tf.app.flags.DEFINE_integer("input_keep_prob", 0.8, "Dropout keep prob for inputs")
tf.app.flags.DEFINE_integer("output_keep_prob", 0.5, "Dropout keep prob for outpus")
tf.app.flags.DEFINE_string("cell", "LSTM", "cell type: LSTM,GRU,LNLSTM")
tf.app.flags.DEFINE_integer("beta_period", 10, "number of epoches before increase Beta.")
#tf.app.flags.DEFINE_integer("beta_period", 1000, "Beta will rise from 0 to 1 during this number of iterations")
#tf.app.flags.DEFINE_integer("beta_offset", 1000, "Beta will start rising after this number of iterations")
tf.app.flags.DEFINE_float("latent_loss_weight", 0.01, "weight used to weaken the latent loss.")
tf.app.flags.DEFINE_integer("dtype_precision", 32, "dtype to be used: typically 32 or 16")
tf.app.flags.DEFINE_boolean("initialize", False, "Initialize model or try to load existing one")
tf.app.flags.DEFINE_string("training_dir" , "auto", "repertory where checkpoints are logs are saved")
FLAGS = tf.app.flags.FLAGS

if FLAGS.training_dir == "auto":
    FLAGS.training_dir = "logs/state"+str(FLAGS.state_size)+"_layers"+str(FLAGS.num_layers)+"_latent"+str(FLAGS.latent_dim)+"_batch"+str(FLAGS.batch_size)+"_"+str(FLAGS.cell)+"_seqs"+str(FLAGS.sequence_min)+"-"+str(FLAGS.sequence_max)+"_"+str(FLAGS.initial_learning_rate)[-1]+"e"+str(int(np.log10(FLAGS.initial_learning_rate)))+"_B"+str(FLAGS.latent_loss_weight)+"_f"+str(FLAGS.dtype_precision)


training_parameters = dict()

if FLAGS.initialize:
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
    training_parameters['learning_rate'] = 2e-4#FLAGS.initial_learning_rate
    

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
batch_gen = Generator(sentences, ratings, FLAGS.batch_size)
num_iters = FLAGS.epoches * batch_gen.iterations_per_epoch()
# deterministic warm-up control
betaGenerator = BetaGenerator(num_iters, FLAGS.beta_period*batch_gen.iterations_per_epoch(), FLAGS.beta_period*batch_gen.iterations_per_epoch())
# text decoder ( text <-> ids)
encoderDecoder = EncoderDecoder()
# learning rate
learningRateControler = LearningRateControler(training_parameters['learning_rate'], FLAGS.learning_rate_change_rate, 0.5)

# load model
vrae_model = Vrae_model(state_size=FLAGS.state_size,
                         num_layers=FLAGS.num_layers,
                          latent_dim=FLAGS.latent_dim,
                         batch_size=FLAGS.batch_size,
                         num_symbols=num_symbols,
                        latent_loss_weight=FLAGS.latent_loss_weight,
                         dtype_precision=FLAGS.dtype_precision,
                        cell_type=FLAGS.cell,
                         input_keep_prob= FLAGS.input_keep_prob,
                        output_keep_prob=FLAGS.input_keep_prob)

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
                padded_batch_xs, batch_ys, batch_lengths, batch_weights, max_length = batch_gen.next_batch()
                training_parameters['learning_rate'] = learningRateControler.learning_rate
                beta = 1#0.01 + betaGenerator(training_parameters['step']) # add small value to points to scatter
                _,d,summary,_ = vrae_model.step(sess, padded_batch_xs, beta, training_parameters['learning_rate'], batch_lengths, batch_weights, training_parameters['epoch'])
                learningRateControler.update(d)
                summary_writer.add_summary(summary, global_step=training_parameters['step'])
                if training_parameters['step'] % 10 == 0:
                    print("loss: " + str(d) + " | step: " + str(training_parameters['step'])  + " | beta: " + str(beta) + " | learning rate: " + str(learningRateControler.learning_rate) )
                training_parameters['step'] += 1 
                
                # increase sentences size
                if training_parameters['n_epoches_since_last_dataset_update'] > 5 and d < FLAGS.acceptable_accuracy and training_parameters['seq_max'] < 50:
                    print "###########################\nUPDATING SENTENCES SIZE"
                    FLAGS.training_dir + '/model'+str(training_parameters['seq_max'])+'.ckp'
                    n_epoches_since_last_dataset_update = 0
                    training_parameters['seq_max'] += 1
                    sentences, ratings = read_data( max_size=None,max_sentence_size=training_parameters['seq_max'],
                                                   min_sentence_size=FLAGS.sequence_min) 
                    batch_gen = Generator(sentences, ratings, FLAGS.batch_size)
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
                