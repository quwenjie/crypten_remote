import torch
import argparse
from mnist_utils import load_plain_feat_label, save_data_each_party
from utils import onehot_tensor

"""
2 users, first party has feature1, second party has feature2
"""


def generate_feature_aggregation_data_config(
    party, data_cnt=100
):
    data_sery = []
    for i in range(data_cnt):
        idx = "MNIST_ID_{}".format(i)
        if party == 0:
            data_record1 = {"ID": idx, "name": "mnist_image1"}
        elif party == 1:
            data_record1 = {"ID": idx, "name": "mnist_image2"}
        data_sery.append(data_record1)
        if party == 0:
            data_sery.append(
                {"ID": idx, "is_feature": False, "name": "classification_label"}
            )
    return data_sery


def split_feature_aggregation_dataset_to_records(data_cnt=100):
    feat_plain, label_plain = load_plain_feat_label()
    feat_list = torch.split(feat_plain, 1)
    feat_list_ret = [
        [feat_list[i][:, 0:15, :].clone(), feat_list[i][:, 15:, :].clone()]
        for i in range(data_cnt)
    ]
    label_list = list(torch.split(label_plain, 1))
    for i in range(len(label_list)):
        if isinstance(label_list[i].item(), int):
            label_list[i] = onehot_tensor([label_list[i].item()]).long()
    return feat_list_ret, label_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run example task")
    parser.add_argument("--party", type=int, help="which party to prepare dataset for")
    args = parser.parse_args()
    data_config=generate_feature_aggregation_data_config(args.party)
    feat_plain, label_plain = split_feature_aggregation_dataset_to_records()
    save_data_each_party(
        p=args.party,
        filename="feature_aggregation_{}.pth",
        feat_plain=feat_plain,
        label_plain=label_plain,
        data_config=data_config,
    )
