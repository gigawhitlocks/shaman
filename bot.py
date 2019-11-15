"""An RNN RocketChat bot -- this file is the entry point
"""
import argparse
import json
import os
import random
import re
import sys
import uuid



import tensorflow as tf
from six import text_type
from six.moves import cPickle
from ssl import SSLError
from websocket import create_connection, WebSocketException

import bottle
from bottle import post, run
from model import Model
from typing import List

import requests
import urllib3

SITE_URL = os.getenv("SHAMAN_SITEURL", "")
BOTNAME = os.getenv("SHAMAN_NAME", "shaman")
TOKEN = os.getenv("SHAMAN_PASSWORD", "shaman")
BOTNAME_NOCASE = re.compile(re.escape(BOTNAME), re.IGNORECASE)

from mattermostdriver import Driver

if SITE_URL == "":
    print("Set SHAMAN_SITEURL")
    sys.exit(1)
if TOKEN == "":
    print("Set SHAMAN_PASSWORD")
    sys.exit(1)

print("Connecting to.. {}".format(SITE_URL))
mm = Driver({
    'url': SITE_URL,
    'token': TOKEN,
    'scheme': 'https',
    'port': 443,
    'verify': True})

mm.login()

# connect to the API
# TODO reimplement for MM

# get room names to id? maybe not necessary for MM
# TODO reimplement for MM
# old implementation:
# ROOMS = {r['name']: r['id'] for r in RC.get_public_rooms()}


def heuristics(saying: str) -> str:
    """Apply rules of thumb to generated text to make it appear more real
    Also clean the output and do things like prevent it from tagging people
    """
    # choose just one thing to say
    saying = random.choice(saying.split("\n"))

    # don't tag people
    saying = saying.replace('@', '@-')

    # don't say bot's own name, triggering more output
    saying = BOTNAME_NOCASE.sub('', saying)

    # strip excess whitespace
    saying = saying.strip()

    return saying

@post('/')
def index():
    """Handles incoming pings.
    """
    print(bottle.request.json)
    inp = bottle.request.json.get('text', '')
    inp = inp.replace(BOTNAME, '').strip()

    if inp == "":
        inp = "I "

    channel = bottle.request.json.get('channel_id', '')

    # for some reason the RNN often just returns spaces
    # for some n calls to sample()
    # so we just call sample() until it outputs anything
    # at all that is not just whitespace
    while True:
        # send 'shaman is typing...'
        # TODO reimplement for MM

        # query some content from the RNN
        saying = sample(inp)

        if inp != "I ":
            # the first line of output includes the input
            # so we need to trim it unless we're
            # prompting shaman to speak in the first person I guess
            saying = saying[len(inp)+1:]

        saying = heuristics(saying)

        if saying == "":
            print("Empty response")

        if saying != "":
            # TODO implement sending the saying
            break

    mm.posts.create_post(options={
        'channel_id': channel,
        'message': saying
    })


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

    init(args)
    run(host='0.0.0.0', port=9871)

def sample(primer):
    """Actually samples the RNN. Uses primer as input. Output is a string."""

    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            return model.sample(sess,
                                chars,
                                vocab,
                                256, primer,
                                args.sample).encode('utf-8').decode('utf8')


def init(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)

    global model
    model = Model(saved_args, training=False)


if __name__ == '__main__':
    main()
