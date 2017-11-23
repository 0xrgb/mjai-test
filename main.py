# Import important libraries
import logging
import mahjong

# Import project pacakges
from ai.random_discard.ai import AI
# TODO: 코드 작성
#import test.hitori

# Before main
Q = logging.getLogger(__name__)
Q.setLevel(logging.INFO)
#Q.setFormat(logging.formatter('%(asctime)s/%(name)-12s/%(levelname)-8s: %(message)s'))

# Main starts
if __name__ == '__main__':
    Q.info('Program start')
    mjai_test(AI) # TODO
    Q.info('Program end')
