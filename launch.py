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
class AliceNet(nn.Module):
    def __init__(self):
        super(AliceNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
 
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

def compute_accuracy(output, labels):
    pred = output.argmax(1)
    correct = pred.eq(labels)
    correct_count = correct.sum(0, keepdim=True).float()
    accuracy = correct_count.mul_(100.0 / output.size(0))
    return accuracy

def encrypt_model_and_data():
    ALICE=0
    BOB=1
    count=100
    crypten.init()
    # Load pre-trained model to Alice
    #sys.modules['AliceNet']=AliceNet
    model = crypten.load_from_party('model.pth', src=ALICE)
    
    # Encrypt model from Alice 
    dummy_input = torch.empty((1, 784))
    private_model = crypten.nn.from_pytorch(model, dummy_input)
    private_model.encrypt(src=ALICE)
    
    # Load data to Bob
    data_enc = crypten.load_from_party('bob_test.pth', src=BOB)
    data_enc2 = data_enc[:count]
    #print(data_enc.size(),data_enc2.size())
    data_flatten = data_enc2.flatten(start_dim=1)
    # Classify the encrypted data
    private_model.eval()
    
    if comm.get().get_rank()==0:
        print('now your turn 0!',)

    if comm.get().get_rank()==1:
        print('now your turn 1!',)

    output_enc = private_model(data_flatten)

    # Compute the accuracy
    output = output_enc.get_plain_text()
    labels = torch.load('testlabel.pth').long()
    accuracy = compute_accuracy(output, labels[:count])
    crypten.print("\tAccuracy: {0:.4f}".format(accuracy.item()),comm.get().get_rank())
    return "\tAccuracy: {0:.4f}".format(accuracy.item())


if __name__ == "__main__":
    print('core addr',file=open('crypten_inference.log','w'))
    print(os.environ['CORE_ADDR'],file=open('crypten_inference.log','a'))
    print(os.environ['JWT'],file=open('crypten_inference.log','a'))
    crypten.common.serial.register_safe_class(AliceNet)
    acc=encrypt_model_and_data()
    print(acc,file=open('crypten_inference.log','a'))

