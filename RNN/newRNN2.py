import argparse
import os
import random
import time

import numpy as np
import tensorflow as tf
import codecs
import math


class ModelNetwork:
    """
    RNN with num_layers LSTM layers and a fully-connected output layer
    The network allows for a dynamic number of iterations, depending on the
    inputs it receives.
       out   (fc layer; out_size)
        ^
       lstm
        ^
       lstm  (lstm size)
        ^
        in   (in_size)
    """
    def __init__(self, in_size, lstm_size, num_layers, out_size, session,
                 learning_rate=0.003, name="rnn"):
        self.scope = name
        self.in_size = in_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.out_size = out_size
        self.session = session
        self.learning_rate = tf.constant(learning_rate)
        # Last state of LSTM, used when running the network in TEST mode
        self.lstm_last_state = np.zeros(
            (self.num_layers * 2 * self.lstm_size,)
        )
        with tf.variable_scope(self.scope):
            # (batch_size, timesteps, in_size)
            self.xinput = tf.placeholder(
                tf.float32,
                shape=(None, None, self.in_size),
                name="xinput"
            )
            self.lstm_init_value = tf.placeholder(
                tf.float32,
                shape=(None, self.num_layers * 2 * self.lstm_size),
                name="lstm_init_value"
            )
            # LSTM
            self.lstm_cells = [
                tf.contrib.rnn.BasicLSTMCell(
                    self.lstm_size,
                    forget_bias=1.0,
                    state_is_tuple=False
                ) for i in range(self.num_layers)
            ]
            self.lstm = tf.contrib.rnn.MultiRNNCell(
                self.lstm_cells,
                state_is_tuple=False
            )
            # Iteratively compute output of recurrent network
            outputs, self.lstm_new_state = tf.nn.dynamic_rnn(
                self.lstm,
                self.xinput,
                initial_state=self.lstm_init_value,
                dtype=tf.float32
            )
            # Linear activation (FC layer on top of the LSTM net)
            self.rnn_out_W = tf.Variable(
                tf.random_normal(
                    (self.lstm_size, self.out_size),
                    stddev=0.01
                )
            )
            self.rnn_out_B = tf.Variable(
                tf.random_normal(
                    (self.out_size,), stddev=0.01
                )
            )
            outputs_reshaped = tf.reshape(outputs, [-1, self.lstm_size])
            network_output = tf.matmul(
                outputs_reshaped,
                self.rnn_out_W
            ) + self.rnn_out_B
            batch_time_shape = tf.shape(outputs)
            self.final_outputs = tf.reshape(
                tf.nn.softmax(network_output),
                (batch_time_shape[0], batch_time_shape[1], self.out_size)
            )
            # Training: provide target outputs for supervised training.
            self.y_batch = tf.placeholder(
                tf.float32,
                (None, None, self.out_size)
            )
            y_batch_long = tf.reshape(self.y_batch, [-1, self.out_size])
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=network_output,
                    labels=y_batch_long
                )
            )
            self.train_op = tf.train.RMSPropOptimizer(
                self.learning_rate,
                0.9
            ).minimize(self.cost)

    # Input: X is a single element, not a list!
    def run_step(self, x, init_zero_state=True):
        # Reset the initial state of the network.
        if init_zero_state:
            init_value = np.zeros((self.num_layers * 2 * self.lstm_size,))
        else:
            init_value = self.lstm_last_state
        out, next_lstm_state = self.session.run(
            [self.final_outputs, self.lstm_new_state],
            feed_dict={
                self.xinput: [x],
                self.lstm_init_value: [init_value]
            }
        )
        self.lstm_last_state = next_lstm_state[0]
        return out[0][0]

    # xbatch must be (batch_size, timesteps, input_size)
    # ybatch must be (batch_size, timesteps, output_size)
    def train_batch(self, xbatch, ybatch):
        init_value = np.zeros(
            (xbatch.shape[0], self.num_layers * 2 * self.lstm_size)
        )
        cost, _ = self.session.run(
            [self.cost, self.train_op],
            feed_dict={
                self.xinput: xbatch,
                self.y_batch: ybatch,
                self.lstm_init_value: init_value
            }
        )
        return cost


def embed_to_vocab(data_, vocab):
    """
    Embed string to character-arrays -- it generates an array len(data)
    x len(vocab).
    Vocab is a list of elements.
    """
    data = np.zeros((len(data_), len(vocab)))
    cnt = 0
    for s in data_:
        v = [0.0] * len(vocab)
        #print(vocab)
        v[vocab.index(s)] = 1.0
        data[cnt, :] = v
        cnt += 1
    return data


def decode_embed(array, vocab):
    return vocab[array.index(1)]


def load_data(input):
    # Load the data
    data_ = ""

    f= codecs.open('f.txt', 'r', encoding='utf8')
    data_ = f.read()

    #data_=data_[0:200000]
    #data_ = data_.lower()
    # Convert to 1-hot coding
    print('popo')
    vocab = sorted(list(set(data_)))
    print('juju')
    print(len(vocab))
    print(len(data_))
    data = embed_to_vocab(data_, vocab)
    return data, vocab


def check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('saved/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)


def main():
   
    parser = argparse.ArgumentParser()
    fileName="f.txt"

    parser.add_argument(
        "--test_prefix",
        type=str,
        default="ব",
        help="Test text prefix to train the network."
    )
    parser.add_argument(
        "--ckpt_file",
        type=str,
        default="saved1/model.ckpt",
        help="Model checkpoint file to load."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=set(("talk", "train")),
        help="Execution mode: talk or train."
    )
    args = parser.parse_args()

    ckpt_file = None
    TEST_PREFIX = args.test_prefix  # Prefix to prompt the network in test mode

    if args.ckpt_file:
        ckpt_file = args.ckpt_file

    # Load the data
    
    data, vocab = load_data(fileName)

    in_size = out_size = len(vocab)
    lstm_size = 1024  # 128
    num_layers = 3
    batch_size = 64  # 128
    time_steps = 100  # 50

    NUM_TRAIN_BATCHES = 50000

    # Number of test characters of text to generate after training the network
    LEN_TEST_TEXT = 10000

    # Initialize the network
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    net = ModelNetwork(
        in_size=in_size,
        lstm_size=lstm_size,
        num_layers=num_layers,
        out_size=out_size,
        session=sess,
        learning_rate=0.003,
        name="char_rnn_network"
    )
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    
    # 1) TRAIN THE NETWORK
    if 1==0:
        
        #check_restore_parameters(sess, saver)
        last_time = time.time()
        batch = np.zeros((batch_size, time_steps, in_size))
        batch_y = np.zeros((batch_size, time_steps, in_size))
        possible_batch_ids = range(data.shape[0] - time_steps - 1)

        print(NUM_TRAIN_BATCHES)
        NUM_TRAIN_BATCHES=10000
        for i in range(NUM_TRAIN_BATCHES):
            # Sample time_steps consecutive samples from the dataset text file
            batch_id = random.sample(possible_batch_ids, batch_size)

            for j in range(time_steps):
                ind1 = [k + j for k in batch_id]
                ind2 = [k + j + 1 for k in batch_id]

                batch[:, j, :] = data[ind1, :]
                batch_y[:, j, :] = data[ind2, :]

            cst = net.train_batch(batch, batch_y)

            if (i % 100) == 0:
                new_time = time.time()
                diff = new_time - last_time
                last_time = new_time
                print("batch: {}  loss: {}  speed: {}  seconds".format(
                    i, cst,  diff
                ))
        #saver.save(sess, "saved/model.ckpt")

        print('yo')
        tf.train.Saver().save(sess, os.path.join('saved1', "model.ckpt"))
    if  1==1:

        saver.restore(sess, ckpt_file)
        f= codecs.open('input.txt', 'r', encoding='utf8')
        datas = f.read()
        datas=datas.replace(chr(2404),'\r\n')
        datas=datas.split('\r\n')
        
        for data_ in datas:
            data_=data_.strip()
            if data_=='':continue
            print(data_)
            numbah=0
            saver.restore(sess, ckpt_file)
            count=0
            for i in range(len(data_)-1):
             
                out = net.run_step(embed_to_vocab(data_[i], vocab), False)
                #print(data_[i])
                for j in range(len(vocab)):
                    k=0
                    if data_[i+1]==vocab[j]:
                        #print(out[j])
                        if out[j]<0.5:
                        
                            k=0
                        else:
                            k=1
                        if out[j]<.002:
                            count=count+1
                        #print(out[j])
                        numbah=numbah+k
                        

            print("Ultimate Value")
            print(numbah/len(data_))
            print(count/len(data_))
            x=numbah/len(data_)
            y=count/len(data_)
            if numbah/len(data_)<.2 or count/len(data_)>.1 and((x-y)>.1):
                print("Invalid")
            else:
                print("Valid")



if __name__ == "__main__":
    main()
