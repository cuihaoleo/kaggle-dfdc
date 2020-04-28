import os
import csv
import shutil
import random

from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import xception_conf as config
from model_def import xception
from augmentation_utils import train_transform, val_transform


def save_checkpoint(path, state_dict, epoch=0, arch="", acc1=0):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if torch.is_tensor(v):
            v = v.cpu()
        new_state_dict[k] = v

    torch.save({
        "epoch": epoch,
        "arch": arch,
        "acc1": acc1,
        "state_dict": new_state_dict,
    }, path)


class DFDCDataset(Dataset):
    def __init__(self, data_csv, required_set, data_root="",
                 ratio=(0.25, 0.05), stable=False, transform=None):
        video_info = []
        data_list = []

        with open(data_csv) as fin:
            reader = csv.DictReader(fin)

            for row in reader:
                if row["set_name"] == required_set:
                    label = int(row["is_fake"])
                    n_frame = int(row["n_frame"])
                    select_frame = round(n_frame * ratio[label])

                    for sample_idx in range(select_frame):
                        data_list.append((len(video_info), sample_idx))

                    video_info.append({
                        "name": row["name"],
                        "label": label,
                        "n_frame": n_frame,
                        "select_frame": select_frame,
                    })

        self.stable = stable
        self.data_root = data_root
        self.video_info = video_info
        self.data_list = data_list
        self.transform = transform

    def __getitem__(self, index):
        video_idx, sample_idx = self.data_list[index]
        info = self.video_info[video_idx]

        if self.stable:
            frame_idx = info["n_frame"] * sample_idx // info["select_frame"]
        else:
            frame_idx = random.randint(0, info["n_frame"] - 1)

        image_path = os.path.join(self.data_root, info["name"],
                                  "%03d.png" % frame_idx)
        try:
            img = Image.open(image_path).convert("RGB")
        except OSError:
            img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)

        if self.transform is not None:
            # img = self.transform(img)
            result = self.transform(image=np.array(img))
            img = result["image"]

        return img, info["label"]

    def __len__(self):
        return len(self.data_list)


def main():
    torch.backends.cudnn.benchmark = True

    train_dataset = DFDCDataset(config.data_list, "train", config.data_root,
                                transform=train_transform)
    val_dataset = DFDCDataset(config.data_list, "val", config.data_root,
                              transform=val_transform, stable=True)

    kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                  shuffle=True, pin_memory=True)
    train_loader = DataLoader(train_dataset, **kwargs)
    val_loader = DataLoader(val_dataset, **kwargs)

    # Model initialization
    model = xception(num_classes=2, pretrained=None)

    if hasattr(config, "resume") and os.path.isfile(config.resume):
        ckpt = torch.load(config.resume, map_location="cpu")
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("acc1", 0.0)
        model.load_state_dict(ckpt["state_dict"])
    else:
        start_epoch = 0
        best_acc = 0.0

    model = model.cuda()
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)

    os.makedirs(config.save_dir, exist_ok=True)

    for epoch in range(config.n_epoches):
        if epoch < start_epoch:
            scheduler.step()
            continue

        print("Epoch {}".format(epoch + 1))

        model.train()

        loss_record = []
        acc_record = []

        for count, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_loss = loss.item()
            loss_record.append(iter_loss)

            preds = torch.argmax(outputs.data, 1)
            iter_acc = torch.sum(preds == labels).item() / len(preds)
            acc_record.append(iter_acc)

            if count and count % 100 == 0:
                print("T-Iter %d: loss=%.4f, acc=%.4f"
                      % (count, iter_loss, iter_acc))

        epoch_loss = np.mean(loss_record)
        epoch_acc = np.mean(acc_record)
        print("Training: loss=%.4f, acc=%.4f" % (epoch_loss, epoch_acc))

        model.eval()
        loss_record = []
        acc_record = []

        with torch.no_grad():
            for count, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                outputs = model(inputs)
                preds = torch.argmax(outputs, 1)

                loss = criterion(outputs, labels)

                iter_loss = loss.item()
                loss_record.append(iter_loss)

                preds = torch.argmax(outputs.data, 1)
                iter_acc = torch.sum(preds == labels).item() / len(preds)
                acc_record.append(iter_acc)

                if count and count % 100 == 0:
                    print("V-Iter %d: loss=%.4f, acc=%.4f"
                          % (count, iter_loss, iter_acc))

            epoch_loss = np.mean(loss_record)
            epoch_acc = np.mean(acc_record)
            print("Validation: loss=%.4f, acc=%.4f" % (epoch_loss, epoch_acc))

            scheduler.step()
            ckpt_path = os.path.join(config.save_dir, "ckpt-%d.pth" % epoch)
            save_checkpoint(
                ckpt_path,
                model.state_dict(),
                epoch=epoch + 1,
                acc1=epoch_acc)

            if epoch_acc > best_acc:
                print("Best accuracy!")
                shutil.copy(ckpt_path,
                            os.path.join(config.save_dir, "best.pth"))
                best_acc = epoch_acc

            print()


if __name__ == "__main__":
    main()
