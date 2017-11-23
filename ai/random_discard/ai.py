import random
from ai.default.ai import AI as defaultAI


class AI(defaultAI):
    def next_move(self, hand, tsumo, remainNumber):
        x = random.randrange(14)
        if x == 13:
            # Tsumogiri
            return tsumo
        else:
            return hand[x]
