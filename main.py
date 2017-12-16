# Import important libraries
import logging
import random
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig

# Import project pacakges
from ai.random_discard.ai import AI

# Before main
logging.getLogger().setLevel(logging.INFO)
#Q.setFormat(logging.formatter('%(asctime)s/%(name)-12s/%(levelname)-8s: %(message)s'))

TEST_NUM = 200    # 테스트 (경기) 횟수
TEST_STEP = 17    # 한 게임이 몇 순인지
TEST_NOTEN_SCORE = -1200   #노텐 = -1200
TEST_TEN_SCORE = +1200  #텐 = +1200
TEST_INIT_PAI = []

def test_ai(ai):
    logging.info("Test start: N = {}, STEP = {}".format(TEST_NUM, TEST_STEP))
    total_score = 0
    for _ in range(TEST_NUM):
        score = test_game(ai)
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
    # 2. tsumo 하자
    for x in range(TEST_STEP):
        # 점수 확인
        result = calculator.estimate_hand_value(hand, tsumo)
        if not result.error:
            return result.cost['main'] * 3
        # 버린다
        discard = ai.next_move(self = ai, hand = hand, tsumo = tsumo, remain_number = TEST_STEP - x - 1)
        hand.remove(discard)
        # 쯔모한다
        tsumo = pai[14 + TEST_STEP]
        hand.append(tsumo)
    # TODO
    return 0

# Main starts
def init():
    global TEST_INIT_PAI
    TEST_INIT_PAI = list(range(136))

if __name__ == '__main__':
    logging.info('Program start')
    init()
    score = test_ai(AI)
    logging.info("Test score: {}".format(score))
    logging.info('Program end')
