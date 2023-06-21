#!/usr/bin/python3
import re
import os
import subprocess

smi = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE)
smi.wait()
smi_out = str(smi.stdout.read())
pids = [int(pid) for pid in re.findall("N/A    (\d+)", smi_out)]
for pid in pids:
    os.kill(pid, 9)
