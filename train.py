import json
import os
import time

import torch
from loguru import logger

from UATVR.network import build_model
from _data_cm import build_loaders, get_topk, get_class_num
from _helper import (
    AverageMeter,
    build_optimizer,
    calc_learnable_params,
    EarlyStopping,
    init,
    print_in_md,
    save_code,
    seed_everything,
    rename_output,
    mean_average_precision,
)
from _helper_cm import evaluate
from config import get_config
from loss import UATVRLoss


def train_epoch(args, dataloader, net, criterion, optimizer, epoch):
    stat_meters = {x: AverageMeter() for x in ["time", "tri_loss", "dsa_loss", "dua_loss", "vib_loss", "loss", "mAP"]}

    toc = time.time()
    net.train()
    for images, texts, labels, _ in dataloader:
        images, texts, labels = images.to(args.device), texts.to(args.device), labels.to(args.device)

        out = net(images, texts, True, True)

        tri_loss, dsa_loss, dua_loss, vib_loss = criterion(*out, labels)
        stat_meters["tri_loss"].update(tri_loss.item())
        stat_meters["dsa_loss"].update(dsa_loss.item())
        stat_meters["dua_loss"].update(dua_loss.item())
        stat_meters["vib_loss"].update(vib_loss.item())

        loss = args.lambda1 * tri_loss + args.lambda2 * dsa_loss + args.lambda3 * dua_loss + args.lambda4 * vib_loss
        stat_meters["loss"].update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stat_meters["time"].update(time.time() - toc)

        # to check overfitting
        map1 = mean_average_precision(out[0].detach().sign(), labels, out[1].detach().sign(), labels).item()
        map2 = mean_average_precision(out[1].detach().sign(), labels, out[0].detach().sign(), labels).item()
        stat_meters["mAP"].update((map1 + map2) / 2)
        toc = time.time()

    info_str = ""
    for x in stat_meters.keys():
        if x == "time":
            info_str += f"[{x}:{stat_meters[x].sum:.2f}s]"
        else:
            info_str += f"[{x}:{stat_meters[x].avg:.4f}]"

    logger.info(
        f"[Training]"
        f"[dataset:{args.dataset}]"
        f"[bits:{args.n_bits}]"
        f"[epoch:{epoch}/{args.n_epochs - 1}]{info_str}"
    )


def train_init(args):
    # setup net
    net = build_model(args)
    # setup criterion
    criterion = UATVRLoss()
    logger.info(f"Number of learnable params: {calc_learnable_params(net)}")
    # setup optimizers
    optimizer = build_optimizer(args.optimizer, net.parameters(), lr=args.lr, weight_decay=args.wd)
    return net, criterion, optimizer


def train(args, train_loader, query_loader, dbase_loader):
    net, criterion, optimizer = train_init(args)
    early_stopping = EarlyStopping()
    for epoch in range(args.n_epochs):
        train_epoch(args, train_loader, net, criterion, optimizer, epoch)
        # monitor mAP every 5 epochs or at the last epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.n_epochs:
            if evaluate(
                args,
                net,
                query_loader,
                dbase_loader,
                early_stopping,
                epoch,
            ):
                break
    logger.info(
        ("No improvements" if early_stopping.counter == early_stopping.patience else "Reach max epoch")
        + f", will save & exit, best mAP: {early_stopping.best_score:.3f}, best epoch: {early_stopping.best_epoch}"
    )
    torch.save(
        early_stopping.best_state, f"{args.save_dir}/e{early_stopping.best_epoch}_{early_stopping.best_score:.3f}.pth"
    )
    return early_stopping.best_epoch, early_stopping.best_score


def main():
    init()
    args = get_config()

    if getattr(args, "rename", False):
        rename_output(args)

    dummy_logger_id = None
    rst = []
    for dataset in ["flickr", "nuswide", "coco", "iapr"]:
        print(f"Processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = build_loaders(args)

        # for hash_bit in [16, 32, 64, 128]:
        for hash_bit in [16, 128]:
            print(f"Processing hash-bit: {hash_bit}")
            seed_everything(args.seed)
            args.n_bits = hash_bit

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=True)
            if any(x.endswith(".pth") for x in os.listdir(args.save_dir)):
                print(f"*.pth exists in {args.save_dir}, will pass...")
                continue

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", mode="w", level="INFO")

            save_code(os.path.dirname(__file__), args.save_dir)
            with open(f"{args.save_dir}/config.json", "w") as f:
                json.dump(vars(args), f, indent=4, sort_keys=True)

            best_epoch, best_score = train(args, train_loader, query_loader, dbase_loader)
            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_epoch": best_epoch, "best_score": best_score})

    print_in_md(rst)


if __name__ == "__main__":
    main()
