import random
from ai.default.ai import AI as defaultAI


class AI(defaultAI):
    def next_move(self, hand, tsumo, remain_number):
        x = random.randrange(14)
        return hand[x]
