import argparse
from os import path as osp


def get_config():
    parser = argparse.ArgumentParser(description=osp.basename(osp.dirname(__file__)))

    # common settings
    parser.add_argument("--backbone", type=str, default="clip", help="see network.py")
    parser.add_argument("--data-dir", type=str, default="../_data_cm", help="directory to dataset")
    parser.add_argument("--n-workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--n-epochs", type=int, default=100, help="number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")
    parser.add_argument("--optimizer", type=str, default="adam", help="sgd/rmsprop/adam/amsgrad/adamw")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="device (accelerator) to use")

    # special settings
    parser.add_argument("--lambda1", type=float, default=1.0, help="weight of sim_loss")
    parser.add_argument("--lambda2", type=float, default=0.05, help="weight of dsa_loss")
    parser.add_argument("--lambda3", type=float, default=0.02, help="weight of dua_loss")
    parser.add_argument("--lambda4", type=float, default=1e-5, help="weight of vib_loss")

    args = parser.parse_args()

    # args.rename = True

    return args
