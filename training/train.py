import os
import time
import sys

import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from training.net import load_model
from training.steerDS import SteerDataSet
from metadata import CLASSES_LIST
from utils.miscellaneous import define_args, get_transforms


def make_weights_for_balanced_classes(labels, nclasses):                        
    count = [0] * nclasses                                                      
    for item in labels:                                                         
        count[item] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(labels)                                              
    for idx, val in enumerate(labels):                                          
        weight[idx] = weight_per_class[val]                                  
    return weight 


def main(args):
    transform = get_transforms(args)

    # Load data
    ds = SteerDataSet(
        os.path.join(os.getcwd(), "data_train"),
        args.crop_ratio,
        transform,
        ".jpg",
    )
    print("The dataset contains %d images " % len(ds))

    trainset, valset = random_split(
        ds, [args.train_split, args.val_split], 
        generator=torch.Generator().manual_seed(args.seed)
    )

    if args.balance_dataset:
        val_list = [ds.filenames[i] for i in valset.indices]
    

    if args.weighted_sampler:
        trainset.dataset.phase = "train"
        temp = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
        labels = []
        for data in temp:
            steering = data["steering"]
            labels += steering.tolist()
    
        weights = make_weights_for_balanced_classes(labels, len(CLASSES_LIST))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     

        trainloader = DataLoader(trainset, batch_size=args.batch_size, 
                                 shuffle=False, sampler=sampler)
    else:
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create network
    net = load_model(args.model_name, args.feat_vect_dim, args.no_global_avg_pool)
    net.to(device)

    # Start from checkpoint if provided
    if args.checkpoint != "":
        path_checkpt = os.path.join(os.getcwd(), args.checkpoint)
        checkpoint = torch.load(path_checkpt)
        net.load_state_dict(checkpoint["model_state_dict"])
        print("Checkpoint loaded")

    # Define loss and optim
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-04)
    else:
        raise NotImplementedError
        
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    phases = ["train", "val"]

    dataloader = {}
    dataloader["train"] = trainloader
    dataloader["val"] = valloader

    writer = SummaryWriter() 
    print("Logging stats at " + writer.log_dir)

    # Train NN
    best_val_acc = 0
    epoch_best_model = None
    for epoch in range(1, args.num_epochs+1, 1):  # loop over the dataset multiple times
        start_time = time.time()
        
        if args.balance_dataset:
            ds_train = SteerDataSet(os.path.join(os.getcwd(), "data_train"),
                                    args.crop_ratio,
                                    transform,
                                    ".jpg",
                                    subsample = True,
                                    val_list = val_list
                                    )
            # Add fake split to have object of the correct class
            trainset, _ = random_split(ds_train, [1.0, 0.0],
                                       generator=torch.Generator().manual_seed(args.seed))
            trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
            dataloader["train"] = trainloader

        loss_epoch = {p: 0 for p in phases}
        accuracy_epoch = {p: [] for p in phases}
        for p in phases:

            is_training = True if p == "train" else False

            net.train(is_training)
            dataloader[p].dataset.dataset.phase = p
            with torch.set_grad_enabled(is_training):

                for i, data in enumerate(dataloader[p], start=1):
                    inputs, labels = data['image'].to(device), data['steering'].to(device)
                    b_size = inputs.shape[0]

                    if is_training:
                        # zero the parameter gradients
                        optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs).squeeze()
                    if b_size == 1:
                        outputs = outputs.unsqueeze(dim=0)
                    loss = criterion(outputs, labels)

                    if is_training:
                        loss.backward()
                        optimizer.step()

                    loss = loss.item()
                    loss_epoch[p] += loss * b_size

                    outputs = outputs.argmax(dim=1)
                    accuracy_epoch[p] += (outputs == labels).tolist()

            writer.add_scalars(
                "Epoch_loss/train_val",
                {p: loss_epoch[p] / len(dataloader[p].dataset)},
                epoch
            )
            writer.add_scalars(
                "Accuracy_loss/train_val",
                {p: sum(accuracy_epoch[p]) / len(accuracy_epoch[p])},
                epoch
            )

        str_format = "[{:<5}]Loss: {:<3.4f}   Accuracy: {:<3.4f}% | "
        log = "Epoch: {:<3}  ".format(epoch)
        for p in phases:
            log += str_format.format(
                p, 
                loss_epoch[p] / len(dataloader[p].dataset),
                sum(accuracy_epoch[p]) / len(accuracy_epoch[p]) * 100,
            )
        log += "Running time: {:<3.1f}s".format(time.time() - start_time)
        print(log)

        if (sum(accuracy_epoch["val"]) / len(accuracy_epoch["val"])) > best_val_acc:
            best_val_acc = sum(accuracy_epoch["val"]) / len(accuracy_epoch["val"])
            epoch_best_model = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "loss_clipped": best_val_acc,
                    "model_state_dict": net.state_dict(),
                    "crop_ratio": args.crop_ratio,
                },
                os.path.join(writer.log_dir, "best_model.pth")
            )
        
        lr_sched.step(loss_epoch['val'])

    # TODO put augmentation e.g. brightness, blur etc
    # TODO decrease learning rate if val loss does not decrease
    # TODO early stopping

    print('Finished Training')
    print("Best validation accuracy is {:.4f} obtained at epoch {}"
          .format(best_val_acc, epoch_best_model))


if __name__ == "__main__":
    launch_dir = os.path.basename(os.getcwd())
    expected = "RVSS_Need4Speed-training"
    assert launch_dir == expected

    args = define_args()

    main(args)
