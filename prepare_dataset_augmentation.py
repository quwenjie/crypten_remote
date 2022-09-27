import torch
import argparse
from mnist_utils import load_plain_feat_label, save_data_each_party
from utils import  onehot_tensor

"""
2 users, each have some data points(feature+label)
"""


def generate_dataset_augmentation_data_config(party, A_data_cnt=100, B_data_cnt=100):
    data_sery = []
    if party == 0:
        L, R = 0, A_data_cnt
    elif party == 1:
        L, R = A_data_cnt, A_data_cnt + B_data_cnt
    for i in range(L, R):
        data_record1 = {"ID": "MNIST_ID_{}".format(i)}
        data_record1["name"] = "mnist_image"
        data_sery.append(data_record1)
        data_record2 = {"ID": "MNIST_ID_{}".format(i)}
        data_record2["name"] = "classification_label"
        data_sery.append(data_record2)
    return data_sery


def split_dataset_augmentation_dataset_to_records(A_data_cnt=100, B_data_cnt=100):
    feat_plain, label_plain = load_plain_feat_label()
    feat_list = torch.split(feat_plain, 1)
    feat_list_ret = [[feat_list[i]] for i in range(A_data_cnt + B_data_cnt)]
    label_list = list(torch.split(label_plain, 1))
    for i in range(len(label_list)):
        if isinstance(label_list[i].item(), int):
            label_list[i] = onehot_tensor([label_list[i].item()]).long()
    return feat_list_ret, label_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run example task")
    parser.add_argument("--party", type=int, help="which party to prepare dataset for")
    args = parser.parse_args()
    data_config = generate_dataset_augmentation_data_config(args.party)
    feat_plain, label_plain = split_dataset_augmentation_dataset_to_records()
    save_data_each_party(
        p=args.party,
        filename="dataset_augmentation_{}.pth",
        feat_plain=feat_plain,
        label_plain=label_plain,
        data_config=data_config,
    )
