import random
import logging
from mahjong.shanten import Shanten

from ai.default.ai import AI as defaultAI
from mahjong.tile import TilesConverter

class AI(defaultAI):
    def __init__(self):
        defaultAI.__init__(self)
        self.shanten = Shanten()
        self.log = logging.getLogger('ShantenAI')

    def next_move(self, hand, tsumo, remain_number):
        min_shanten = 8
        check = [False] * 34 # 핸드 중복 확인
        check_shanten = [8] * 14  # 손에 있는 14개 중에 어떤 것이 샹텐이 작은지 확인하는 배열

        hand_34 = TilesConverter.to_34_array(hand)

        for x in range(len(hand)):
            x4 = hand[x] // 4
            if check[x4]: # 같은 타일이면 체크 안해도 됨
                continue
            check[x4] = True

            hand_34[x4] -= 1
            max_shanten = -2
            for y in range(34):
                if y == x4 or hand_34[y] == 4:
                    continue
                hand_34[y] += 1
                result = self.shanten.calculate_shanten(hand_34)
                if result > max_shanten:
                    max_shanten = result
                hand_34[y] -= 1
            hand_34[x4] += 1
            if min_shanten > max_shanten:
                min_shanten = max_shanten
            check_shanten[x] = max_shanten

        check_discard = [] #무엇을 버려야 할지 결정해주는 배열
        for x in range(14):
            if check_shanten[x] == min_shanten: #최소 샨텐에 해당하는 번째의 패이면
                check_discard.append(x)

        self.log.debug("#: %d", min_shanten)
        return hand[random.choice(check_discard)]

    def get_name(self):
        return "shanten_min"
