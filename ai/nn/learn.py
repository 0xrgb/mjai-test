import numpy as np
import random
import math
import logging
import tensorflow as tf


class CatchEnvironment:
    def __init__(self):
        #self.__init__()
        self.log = logging.getLogger('Learning Session')

    def observe(self, hand): # 손의 상태를 알려주는
        hand_copy_136 = hand.copy()
        self.state = hand_copy_136
        return self.state

    #def drawState(self):
    #def reset(self, hand):
    #def getState(self):
    #def getReward(self):
    #def isGameOver(self):
    #def updateState(self, action):

    #def act(self, action):
        #self.updateState(action)
        #reward = self.getReward()
        #gameOver = self.isGameOver()
        #return self.observe(), reward, gameOver, self.getState()


class ReplayMemory:
    def __init__(self, nbActions, nbStates, maxMemory, discount):
        self.maxMemory = maxMemory
        self.nbStates = nbStates
        self.discount = discount
        self.inputState = np.empty((self.maxMemory, nbActions), dtype=np.uint8) #TODO(float32 -> uint8로 변경해도 되는지 확인)
        self.actions = np.zeros(self.maxMemory, dtype=np.uint8)
        self.nextState = np.empty((self.maxMemory, nbActions), dtype=np.uint8) #TODO(float32 -> uint8로 변경해도 되는지 확인)
        self.gameOver = np.empty(self.maxMemory, dtype=np.bool)
        self.rewards = np.empty(self.maxMemory, dtype=np.int32) #TODO(int8 -> int32로 변경해도 되는지 확인)
        self.count = 0
        self.current = 0

    def remember(self, currentState, action, reward, nextState, gameOver):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.inputState[self.current, ...] = currentState
        self.nextState[self.current, ...] = nextState
        self.gameOver[self.current] = gameOver
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.maxMemory

    def getBatch(self, model, batchSize, nbActions, nbStates, sess, X):
        memoryLength = self.count
        chosenBatchSize = min(batchSize, memoryLength)

        inputs = np.zeros((chosenBatchSize, nbStates))
        targets = np.zeros((chosenBatchSize, nbActions))

        # 입력과 타겟을 체워주는 과정
        for x in range(chosenBatchSize):
            if memoryLength == 1:
                memoryLength = 2
            # 랜덤한 경험을 골라서 batch에 추가한다.
            randomIndex = random.randrange(1, memoryLength)
            current_inputState = np.reshape(self.inputState[randomIndex], (1, nbActions))

            target = sess.run(model, feed_dict={X:current_inputState})

            current_nextState = np.reshape(self.nextState[randomIndex], (1, nbActions))
            current_outputs = sess.run(model, feed_dict={X: current_nextState})

            # Gives us Q_sa, the max q for the next state.
            nextStateMaxQ = np.amax(current_outputs)
            if (self.gameOver[randomIndex] == True):
                target[0, [self.actions[randomIndex] - 1]] = self.rewards[randomIndex]
            else:
                # reward + discount(gamma) * max_a' Q(s',a')
                # We are setting the Q-value for the action to  r + gamma*max a' Q(s', a'). The rest stay the same
                # to give an error of 0 for those outputs.
                target[0, [self.actions[randomIndex] - 1]] = self.rewards[randomIndex] + self.discount * nextStateMaxQ

            # Update the inputs and targets.
            inputs[x] = current_inputState
            targets[x] = target

        return inputs, targets

    def Reward_Handling(self, currentState, action, reward, nextState, gameOver, env, err, epsilon, epsilonMinimumValue, output_layer, batchSize, nbActions, nbStates, X, Y, optimizer, cost ):
        with tf.Session() as sess:

            # 입실론 0.999 만큼 곱해가면서 점점 줄여나가기
            if epsilon > epsilonMinimumValue:
                epsilon = epsilon * 0.999

            #nextState, reward, gameOver, stateInfo = env.act(action) #삭제해야 할꺼 같은 부분

            self.remember(currentState, action, reward, nextState, gameOver)  # TODO 일단 여기

            ''' # 이미 업데이트 해서 가져 놓은거라 필요 없음
            # Update the current state and if the game is over.
            currentState = nextState
            isGameOver = gameOver
            '''

            # We get a batch of training data to train the model.
            inputs, targets = self.getBatch(output_layer, batchSize, nbActions, nbStates, sess, X)

            # Train the network which returns the error.
            _, loss = sess.run([optimizer, cost], feed_dict={X: inputs, Y: targets})
            err = err + loss