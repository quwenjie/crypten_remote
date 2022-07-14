# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import logging
import os
import crypten
import crypten.communicator as comm 

import torch



if __name__ == "__main__":
    from dnn import encrypt_model_and_data,AliceNet
    crypten.common.serial.register_safe_class(AliceNet)
    encrypt_model_and_data()
