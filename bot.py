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
from rocketchat.api import RocketChatAPI
from rocketchat.calls.auth.get_me import GetMe
from six import text_type
from six.moves import cPickle
from websocket import create_connection

import bottle
from bottle import post, run
from model import Model

SITE_URL = os.getenv("SHAMAN_SITEURL", "")
BOTNAME = os.getenv("SHAMAN_NAME", "shaman")
BOTPASSWORD = os.getenv("SHAMAN_PASSWORD", "shaman")
BOTNAME_NOCASE = re.compile(re.escape(BOTNAME), re.IGNORECASE)

if SITE_URL == "":
    print("Set SHAMAN_SITEURL")
    sys.exit(1)
if BOTPASSWORD == "":
    print("Set SHAMAN_PASSWORD")
    sys.exit(1)

print("Connecting to.. {}".format(SITE_URL))
RC = RocketChatAPI(settings={'username': BOTNAME, 'password': BOTPASSWORD,
                             'domain': 'https://' + SITE_URL})

ROOMS = {r['name']: r['id'] for r in RC.get_public_rooms()}

@post('/')
def index():
    """Handles incoming pings.
    Main entrypoint from Rocket Chat.
    """

    inp = bottle.request.json.get('text', '')
    inp = inp.replace(BOTNAME, '')
    if inp == "":
        inp = "I "

    # get an auth token from the rest api
    auth_token = GetMe(RC.settings).auth_token
    url = "wss://" + SITE_URL + "/websocket"

    ws = create_connection(url) # pylint: invalid-name
    trace = uuid.uuid4().hex[:5]
    # must ping first
    ws.send(json.dumps({
        "msg": "connect",
        "version": "1",
        "support": ["1"]
    }))

    # login
    ws.send(json.dumps({
        "msg": "method",
        "method": "login",
        "id": trace,
        "params": [
            {"resume": auth_token}
        ]
    }))

    channel = bottle.request.json.get('channel_name', '')

    # for some reason the RNN often just returns spaces
    # for some n calls to sample()
    # so we just call sample() until it outputs anything
    # at all that is not just whitespace
    while True:
        # send 'shaman is typing...'
        ws.send(json.dumps({
            "msg": "method",
            "method": "stream-notify-room",
            "id": trace,
            "params": [
                ROOMS[channel]+"/typing",
                BOTNAME,
                True
                ]
            }))

        # query some content from the RNN
        saying = sample(inp)

        def heuristics(saying: str) -> str:
            """Apply rules of thumb to generated text to make it appear more real
            Also clean the output and do things like prevent it from tagging people
            """
            saying = saying.split("\n")
            idx = random.randint(0, len(saying)-1)

            saying = saying[idx]

            # the first line of output includes the input
            # so we need to trim it
            if idx == 0 and inp != "I ":
                saying = saying[len(inp)+1:]

            # don't tag people
            saying = saying.replace('@', '@-')

            # don't say bot's own name, triggering more output
            saying = BOTNAME_NOCASE.sub('', saying)

            # strip excess whitespace
            saying = saying.strip()

            return saying

        saying = heuristics(saying)

        if saying != "":
            # say it
            RC.send_message(saying, channel)

            # stop typing
            ws.send(json.dumps({
                "msg": "method",
                "method": "stream-notify-room",
                "id": trace,
                "params": [
                    ROOMS[channel]+"/typing",
                    "shaman",
                    False
                ]
            }))

            ws.close()
            break


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
    run(host='0.0.0.0', port=9001)

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
                                500, primer,
                                args.sample).encode('utf-8').decode('utf8')


def init(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)

    global model
    model = Model(saved_args, training=False)

if __name__ == '__main__':
    main()
