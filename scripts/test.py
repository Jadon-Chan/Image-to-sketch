import subprocess
import os

home = os.environ["HOME"]

output = subprocess.check_output(f"{home}/img2img/scripts/find_avail.sh")

num, avail = output.split()
num = int(num)
avail = int(avail)
print(num, avail)

