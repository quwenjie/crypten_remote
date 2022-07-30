# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import crypten
import torch.nn as nn
import torch.nn.functional as F
import sys
import crypten.communicator as comm
import time
import argparse
import logging
import os
import json
import torchvision
from colink.sdk_a import CoLink, byte_to_str, StorageEntry


class MLP(nn.Module):
    def __init__(self, layer_emb, activation):
        super().__init__()
        self.layer_cnt=0
        
        for i in range(len(layer_emb) - 1):
            setattr(self,'layer{}'.format(self.layer_cnt),nn.Linear(layer_emb[i], layer_emb[i + 1]))
            self.layer_cnt+=1
            if i != len(layer_emb) - 2:
                if activation == "relu":
                    setattr(self,'layer{}'.format(self.layer_cnt),nn.ReLU())
                    self.layer_cnt+=1

    def forward(self, input):
        x=input
        for i in range(self.layer_cnt):
            layer=getattr(self,'layer{}'.format(i))
            x=layer(x)
        return x


def download_online_file(path):
    basename = os.path.basename(path)
    if os.path.exists(basename):  # remove repeated file
        os.remove(basename)
    os.system("wget {}".format(path))
    return basename


def load_from_location(loc, src):
    if loc["type"] == "online":
        basename=""
        #if comm.get().get_rank()==src:
        basename = download_online_file(loc["path"])
        data = crypten.load_from_party(basename, src=src)
        return data
    else:
        print("location type {} not supported!".format(loc["type"]))
    return 


def compute_accuracy(output, labels):
    pred = output.argmax(1)
    correct = pred.eq(labels)
    correct_count = correct.sum(0, keepdim=True).float()
    accuracy = correct_count.mul_(100.0 / output.size(0))
    return accuracy


if __name__ == "__main__":
    """
    print("core addr", file=open("crypten_inference.log", "w"))
    core_addr = os.environ["CORE_ADDR"]
    jwt = os.environ["JWT"]
    app_id = os.environ["CRYPTEN_APP_ID"]
    cl = CoLink(core_addr, jwt)
    res = cl.read_entries(
        [
            StorageEntry(
                key_name="crypten:{}:config".format(app_id),
            )
        ]
    )
    if res is not None:
        json_str = byte_to_str(res[0].payload)
        print(json_str, file=open("crypten_inference.log", "a"))
    else:
        print("read json in storage failure", file=open("crypten_inference.log", "a"))
        sys.exit()
    """
    json_str = open("1.json", "r", encoding="utf8").read()
    json_data = json.loads(json_str)
    architecture_config = json_data["architecture"]
    inference_config = json_data["inference"]
    transform_config = json_data["transform"]
    dataset_config=json_data["dataset"]
    model_config = json_data["model"]
    input_shape=json_data['input_shape']
    
    ALICE = 0
    BOB = 1
    crypten.init()

    rank = comm.get().get_rank()

    if architecture_config["type"] == "pytorch-builtin":
        arch_arg = architecture_config["construct_arg"]
        model_class = getattr(torchvision.models, architecture_config["name"])
        model = model_class(**arch_arg)

    elif architecture_config["type"] == "custom":
        arch_arg = architecture_config["construct_arg"]
        model_class = getattr(sys.modules[__name__], architecture_config["name"])
        model = model_class(**arch_arg)
    
    crypten.common.serial.register_safe_class(model_class)

    if model_config["pretrained"] == True:  # need to download file
        loc = model_config["location"]
        model = load_from_location(loc, src=ALICE)
    
    print('alice here!')
    

    dataset_loc=dataset_config["location"]
    data_enc = load_from_location(dataset_loc, src=BOB)
    print('bob xxx')
    data_dec = data_enc.get_plain_text()
    data_dec = data_dec[: inference_config["inference_number"]]
    for i in range(len(transform_config)):
        trans_config = transform_config[i]["construct_arg"]
        trans_class = getattr(torchvision.transforms, transform_config[i]["type"])
        trans = trans_class(**trans_config)
        data_dec = trans(data_dec)
    data_dec=data_dec.reshape([-1]+input_shape[1:])
    print('fuck!')
    input_data=crypten.cryptensor(data_dec)
    print('here end!')
    
    dummy_input = torch.empty(input_shape)
    private_model = crypten.nn.from_pytorch(model, dummy_input)
    private_model.encrypt(src=ALICE)
    private_model.eval()
    print('enc end!')

    
    batch_size=inference_config["batch_size"]
    output=[]

    for b in range((input_data.size(0)-1)//batch_size+1):
        input_batch=input_data[b*batch_size:(b+1)*batch_size]
        out=private_model(input_batch)
        output.append(out)
    output=crypten.cat(output,dim=0)
    print(output.size())


    print(acc, file=open("crypten_inference.log", "a"))
