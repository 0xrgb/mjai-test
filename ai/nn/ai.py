import numpy as np
import random
import math
import logging
import tensorflow as tf

from ai.default.ai import AI as defaultAI
from ai.nn.learn import CatchEnvironment
from ai.nn.learn import ReplayMemory

# a, b 사이의 랜덤 값 출력
def randf(a, b):
    return (float(random.randrange(0,(b - a) * 9999)) / 10000) + a

class AI(defaultAI):
    def __init__(self):
        defaultAI.__init__(self)
        self.log = logging.getLogger('NN_AI')

    def next_move(self, hand, tsumo, remain_number, env, X, output_layer, nbActions, isGameOver, epsilon):
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            currentState = env.observe(hand)
            action = -9999  # action initilization

            # 이제 랜덤으로 액션을 결정할지 아니면 policy network에서 액션을 결정할지 정함
            if randf(0, 1) <= epsilon:  # 랜덤 액션 결정 #TODO 수치 확인
                action = random.randrange(0, nbActions)
            else:
                q = sess.run(output_layer, feed_dict={X:currentState})
                # 최대치 가지는 index 선택
                index = q.argmax()
                action = index
            return action
        #self.log.debug()

    def get_name(self):
        return "nn_ai"




























