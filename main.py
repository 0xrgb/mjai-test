# Import important libraries
import logging
import random
import math
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig
from mahjong.tile import TilesConverter
from mahjong.shanten import Shanten

# Import project pacakges
#from ai.random_discard.ai import AI     # 랜덤 AI할때 사용하기
#from ai.shanten.ai import AI            # 샹텐 AI할때 사용하기

#nn_ai 할때 사용하는 팩키지 모음 시작
from ai.nn.ai import AI
import tensorflow as tf
from ai.nn.learn import CatchEnvironment
from ai.nn.learn import ReplayMemory
#모음 끝

TEST_NUM = 500    # 테스트 (경기) 횟수
TEST_STEP = 17    # 한 게임이 몇 순인지
TEST_NOTEN_SCORE = -1200   #노텐 = -1200
TEST_TEN_SCORE = +1200  #텐 = +1200
TEST_INIT_PAI = []


#nn_ai 할때 사용할 변수, 베이스 구조 목록 시작

epsilon = 1  # 학습 하는 동안에 랜덤 행위를 할 확률 (범위 0 - 1) 학습할 수록 이 값은 줄어든다
epsilonMinimumValue = 0.001  # 입실론이 가기를 원하는 최소값 (범위 0 - 1)
nbActions = 14  # 할 수 있는 행동의 가짓수 14개의 패중에서 버리는 것이기에 14로 설정
#epoch
hiddenSize = 100  # hidden layers 에 있는 뉴런 개수
maxMemory = 136  # 메모리가 얼마나 많아야되는지
batchSize = 50  # The mini-batch size for training. Samples are randomly taken from memory till mini-batch size.
#gridSize 필요 없을꺼 같아서 안 넣음
nbStates = 34 * 4  # We eventually flatten to a 1d tensor to feed the network.
discount = 0.9  # The discount is used to force the network to choose states that lead to the reward quicker (0 to 1)
learningRate = 0.2  # Learning Rate for Stochastic Gradient Descent (our optimizer).

err = 0

# 베이스 구조 만들기
X = tf.placeholder(tf.float32, [None,nbStates]) # 입력값 #TODO(float32 -> 다른data type로 변경해도 되는지 확인)
#TODO 여기부터 밑에 부분 tf.truncated_normal 쓴 것이 맞는지 확인하기
W1 = tf.Variable(tf.truncated_normal([nbStates, hiddenSize], stddev=1.0 / math.sqrt(float(nbStates))))
b1 = tf.Variable(tf.truncated_normal([hiddenSize], stddev=0.01))
input_layer = tf.nn.relu(tf.matmul(X, W1) + b1)
W2 = tf.Variable(tf.truncated_normal([hiddenSize, hiddenSize], stddev=1.0 / math.sqrt(float(hiddenSize))))
b2 = tf.Variable(tf.truncated_normal([hiddenSize], stddev=0.01))
hidden_layer = tf.nn.relu(tf.matmul(input_layer, W2) + b2)
W3 = tf.Variable(tf.truncated_normal([hiddenSize, nbActions], stddev=1.0 / math.sqrt(float(hiddenSize))))
b3 = tf.Variable(tf.truncated_normal([nbActions], stddev=0.01))
output_layer = tf.matmul(hidden_layer, W3) + b3

Y = tf.placeholder(tf.float32, [None, nbActions]) # 출력값  #TODO(float32 -> )

# Mean squared error cost function
cost = tf.reduce_sum(tf.square(Y - output_layer)) / (2 * batchSize)

# Stochastic Gradient Decent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)


#목록 끝


def test_ai(ai):
    logging.info("Test: AI = %s, N = %d, STEP = %d", ai.get_name(), TEST_NUM, TEST_STEP)
    total_score = 0
    for _ in range(1, TEST_NUM + 1):
        score = test_game(ai)
        logging.info("Game %03d / Score: %s", _, score)
        total_score += score
    return total_score / TEST_NUM

def test_game(ai):
    pai = TEST_INIT_PAI.copy()
    random.shuffle(pai)

    calculator = HandCalculator()
    config = HandConfig(is_tsumo=True)
    config.yaku.yakuhai_place = config.yaku.east
    config.yaku.yakuhai_round = config.yaku.east

    # 1. 내 초기 패 13개 + 쯔모 1개
    hand = pai[:14].copy()
    tsumo = pai[13]


    # 1 - a.  nn_ai를 돌리는 경우에 env에 정보를 입력한다.
    err = 0
    isGameOver = False


    # 2. tsumo 하자
    for x in range(1, TEST_STEP + 1):

        # DEBUG: 정상 작동하는지 중간 패 과정 출력
        logging.debug("%02d: %s", x, TilesConverter.to_one_line_string(hand))
        # 점수 확인
        result = calculator.estimate_hand_value(hand, tsumo)
        if not result.error:
            # 화료가 되었다는 뜻이다. 친의 쯔모인 3배 점수를 반환하자
            isGameOver = True
            return result.cost['main'] * 3

        if x == TEST_STEP:
            break
        '''
        # 여기부터
        # 버린다
        discard = ai.next_move(hand, tsumo, TEST_STEP - x - 1)  # 일반적인 경우 이거 쓰기
        hand.remove(discard)
        # 쯔모한다
        tsumo = pai[14 + x]
        hand.append(tsumo)
        # 여기까지는 nn_ai 아닐때 쓰는 코드
        '''

        #여기부터
        discard_action = ai.next_move(hand, tsumo, TEST_STEP - x - 1, env, X, output_layer, nbActions, isGameOver, epsilon)  #nn_ai이면 이 코드로 변경
        discard = hand[discard_action]
        currentState = hand.copy()
        hand.remove(discard)
        tsumo = pai[14 + x]
        hand.append(tsumo)
        nextState = hand.copy()
        result = calculator.estimate_hand_value(hand, tsumo)
        if not result.error:
            isGameOver = True
            reward = result.cost['main'] * 3
        else:
            hand_34 = TilesConverter.to_34_array(hand)
            shanten = Shanten()
            result = shanten.calculate_shanten(hand_34)
            logging.debug('@: %d', result)
            if result == 0:
                reward = TEST_TEN_SCORE
            reward = TEST_NOTEN_SCORE
        memory.Reward_Handling(currentState, discard_action, reward, nextState, isGameOver, env, err, epsilon, epsilonMinimumValue, output_layer, batchSize, nbActions, nbStates, X, Y, optimizer, cost)
        #여기까지는 nn_ai쓸때만 쓰는 코드


    # 마지막으로 텐파이인지 확인한다
    # 샹텐수가 0이면 텐파이이다
    hand_34 = TilesConverter.to_34_array(hand)
    shanten = Shanten()
    result = shanten.calculate_shanten(hand_34)
    logging.debug('@: %d', result)
    if result == 0:
        return TEST_TEN_SCORE
    return TEST_NOTEN_SCORE

# Main starts
def init():
    global TEST_INIT_PAI
    TEST_INIT_PAI = list(range(136))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    init()
    logging.info('Program start')


    #nn_ai 테스트 이외의 테스트 시 주석 처리해야 됨 (시작부분)

    env = CatchEnvironment()
    memory = ReplayMemory(nbActions, nbStates, maxMemory, discount)
    # Add ops to sae and restore all the variables
    #saver = tf.train.Saver() #저장 부분은 잠시 따로
    #tf.initialize_all_variables.run()
    #tf.initialize_all_variables()
    tf.global_variables_initializer()

    #nn_ai 테스트 이외의 테스트 시 주석 처리 해야됨 (끝 부분)


    score = test_ai(AI())
    logging.info("Average score: {}".format(score))
    logging.info('Program end')

    ''' #변수 저장할때 
    # Save the variables to disk.
    save_path = saver.save(sess, os.getcwd() + "/model.ckpt")
    print("Model saved in file: %s" % save_path)
    '''
