import torch
import sys
from torch import device
from glob import glob


def parse_file(fn):
    objects = {}
    with open(fn, 'r') as f:
        content = f.read().split("\n")
    for line in content:
        if len(line) <= 10 or "====" in line or "|" in line or "----" in line:
            continue
        objname, res = line.split(", shape:")
        objname = objname[21:-2]
        device_dict = eval(res.split("device: ")[-1])
        objects[objname] = device_dict
    return objects
    
if __name__ == "__main__":
    objlist = []
    for fn in glob("gpuusage/*.txt"):
        objlist.append(parse_file(fn))
    deepspeed = "deepspeed.runtime.engine.DeepSpeedEngine"
    deepspeed_list = [obj[deepspeed] for obj in objlist]
    keylists = []
    vallists = []
    for dev_layers in deepspeed_list:
        # dev_layers: {layername: (layer_shape, device_of_this_layer), }
        keylists.append(list(dev_layers.keys()))  # layer name
        vallists.append([i[0] for i in list(dev_layers.values())] )  # layer shape
        # for key in dev_layers:
        #     print(key, dev_layers[key])
        # print("="*80)
    for i in range(len(keylists)-1):
        print(keylists[i] == keylists[i+1])
    for i in range(len(vallists)-1):
        print(vallists[i] == vallists[i+1])

