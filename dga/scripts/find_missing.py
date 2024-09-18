import subprocess
import sys
import os

temp = sys.argv[1]
run = sys.argv[2]

homedir = "/project/dinner/scguo/kaiB/dga/logs"

files = subprocess.check_output(f'grep -rl "STEP" {homedir}/{temp}/{run}/*.err', shell=True)
filestr = []
for f in files:
    filestr.append(f.decode().split("/")[-1].split(".")[0])
with open(f"missing_{temp}_{run}.txt", mode='w') as f:
    f.write(','.join(filestr[:-1]))
