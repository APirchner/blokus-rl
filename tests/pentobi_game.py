"""Records a Classic game that Pentobi plays against itself.

Writes one line per move, the color (1 to 4) and the move, then the
final score. The Rust test `test_pentobi_game_replays` replays it.

Usage: python tests/pentobi_game.py PENTOBI_GTP > tests/pentobi_game.txt
"""

import subprocess
import sys


def main(pentobi_gtp: str) -> None:
    engine = subprocess.Popen(
        [pentobi_gtp, "--game", "classic", "--nobook", "--quiet"]
        + ["--level", "3", "--seed", "1", "--threads", "1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    def send(command: str) -> str:
        engine.stdin.write(command + "\n")
        engine.stdin.flush()
        # a response ends with an empty line, readline() returns "" once Pentobi exits
        response = ""
        while (line := engine.stdout.readline()) != "\n":
            assert line, f"{command}: Pentobi exited"
            response += line
        response = response.strip()
        assert response.startswith("="), f"{command}: {response}"
        return response[1:].strip()

    # a color out of moves passes, the game ends when all of them pass
    color, passes = 1, 0
    while passes < 4:
        move = send(f"genmove {color}")
        print(color, move)
        passes = passes + 1 if move == "pass" else 0
        color = color % 4 + 1
    print("final_score", send("final_score"))
    send("quit")


if __name__ == "__main__":
    main(sys.argv[1])
