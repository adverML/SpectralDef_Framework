
#!/usr/bin/env python3

def get_dataloader(dataset_type, root_dir, is_train, batch_size, workers, resolution=32, classes=1000, **kwargs):
    # normalize = transforms.Normalize(mean=kwargs["mean"], std=kwargs["std"])
    normalize = transforms.Normalize(mean=[0.4810, 0.4574, 0.4078], std=[0.2146, 0.2104, 0.2138])

    transformations = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ] if is_train else [
        transforms.ToTensor(),
        normalize,
    ]
    trans = transforms.Compose(transformations)
    dataset = smallimagenet.SmallImagenet(root=root_dir, size=resolution, train=is_train, transform=trans,
                                          classes=range(classes)) if dataset_type == "SmallImageNet" else tinyimagenet.TinyImageNet(
        root=root_dir, train=is_train, transform=trans)
    shuffle = True # if is_train else False
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=True)
    return loader