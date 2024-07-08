# coding : utf-8
# Author : yuxiang Zeng
import platform
import multiprocessing
import dgl
from torch.utils.data import DataLoader



def custom_collate_fn(batch, args):
    from torch.utils.data.dataloader import default_collate
    userIdx, servIdx, values = zip(*batch)
    userIdx, servIdx = default_collate(userIdx), default_collate(servIdx)
    values = default_collate(values)
    return userIdx, servIdx, values


def get_dataloaders(train_set, valid_set, test_set, args):
    # max_workers = multiprocessing.cpu_count()
    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, args),
        # num_workers=max_workers,
        # prefetch_factor=4
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=4096,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, args),
        # num_workers=max_workers,
        # prefetch_factor=4
    )
    test_loader = DataLoader(
        test_set,
        batch_size=4096,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, args),
        # num_workers=max_workers,
        # prefetch_factor=4
    )

    return train_loader, valid_loader, test_loader



if __name__ == '__main__':
    print(multiprocessing.cpu_count())
