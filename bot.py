from __future__ import print_function
import tensorflow as tf

import argparse
import os
from six.moves import cPickle

from model import Model

from six import text_type

from rocketchat.api import RocketChatAPI
rc = RocketChatAPI(settings={'username': '', 'password': '',
                              'domain': ''})
from bottle import post, run, template, request
import bottle
import random
import re

@post('/')
def index():
    sayings = sample(bottle.request.json.get('text', '')).split("\n")
    sayings = sayings[6:]
    thefirstone = False
    for saying in sayings:
        if not thefirstone :
            thefirstone = True
            continue

        saying.replace('@', '@-')
        if len(saying) == 0 :
            continue

        rc.send_message(saying, "GENERAL")
        if random.randint(0,9) < 5:
            break
    return ''


def main():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                        help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u' ',
                        help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')

    global args
    args = parser.parse_args()

    model = init(args)
    run(host='0.0.0.0', port=9001)

def sample(primer):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            return model.sample(sess, chars, vocab, args.n, primer,
                                args.sample).encode('utf-8').decode('utf8')


def init(args):

    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    global model
    model = Model(saved_args, training=False)

if __name__ == '__main__':
    main()
