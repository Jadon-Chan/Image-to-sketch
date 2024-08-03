import os
from subprocess import check_output

from typing import Tuple

home = os.environ['HOME']

def check_dcu_status() -> Tuple[bool, int]:
    output = check_output(f"./scripts/find_avail.sh")
    num, avail = output.split()
    num, avail = int(num), int(avail)
    return (num!=0, avail)
