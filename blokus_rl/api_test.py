from pettingzoo.test import api_test
from blokus_rl.blokus import BlokusEnv

if __name__ == "__main__":
    api_test(BlokusEnv(), num_cycles=10, verbose_progress=True)
