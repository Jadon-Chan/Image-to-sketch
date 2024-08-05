import os
from subprocess import check_output, run, Popen

from typing import Tuple

def check_dcu_status() -> Tuple[bool, int]:
    output = check_output(f"scripts/find_avail.sh")
    num, avail = output.split()
    num, avail = int(num), int(avail)
    return (num!=0, avail)

def next_id() -> int:
    files = os.listdir("./inits")
    return len(files)

def infer(id: int):
    Popen(['python', 'scripts/anime_from_image.py', '--init_image_name', f"{id}.png"])
