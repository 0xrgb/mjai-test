# Import important libraries
import logging
import random
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig
from mahjong.tile import TilesConverter
from mahjong.shanten import Shanten

# Import project pacakges
#from ai.random_discard.ai import AI
from ai.shanten.ai import AI

TEST_NUM = 100    # 테스트 (경기) 횟수
TEST_STEP = 17    # 한 게임이 몇 순인지
TEST_NOTEN_SCORE = -1200   #노텐 = -1200
TEST_TEN_SCORE = +1200  #텐 = +1200
TEST_INIT_PAI = []

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
    # 2. tsumo 하자
    for x in range(1, TEST_STEP + 1):
        # DEBUG: 정상 작동하는지 중간 패 과정 출력
        logging.debug("%02d: %s", x, TilesConverter.to_one_line_string(hand))
        # 점수 확인
        result = calculator.estimate_hand_value(hand, tsumo)
        if not result.error:
            # 화료가 되었다는 뜻이다. 친의 쯔모인 3배 점수를 반환하자
            return result.cost['main'] * 3

        if x == TEST_STEP:
            break
        # 버린다
        discard = ai.next_move(hand, tsumo, TEST_STEP - x - 1)
        hand.remove(discard)
        # 쯔모한다
        tsumo = pai[14 + x]
        hand.append(tsumo)
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
    score = test_ai(AI())
    logging.info("Average score: {}".format(score))
    logging.info('Program end')
