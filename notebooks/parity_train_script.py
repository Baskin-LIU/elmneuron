# Imports
import gc
import json
import os
import random
import sys
import tempfile
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Relative Imports
package_path = Path(os.path.abspath(os.path.join(os.path.dirname('__file__'), '..')))  # TODO: change to elmneuron path
sys.path.insert(0, str(package_path))

from src.expressive_leaky_memory_neuron_v2 import ELM
from src.expressive_leaky_memory_neuron_forget import ELMf
from src.parity_tasks import make_batch_Nbit_pair_parity, make_batch_Nbit_pair_paritysum

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--short_run", dest="short_run", action="store_true")
    parser.add_argument("--forget_gate", dest="forget_gate", action="store_true")
    parser.add_argument("--learn_mem_tau", dest="learn_mem_tau", action="store_true")
    parser.add_argument("--curriculum", dest="curriculum", action="store_true")
    parser.add_argument("--Nsum", dest="Nsum", action="store_true")
    
    
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--num_memory", type=int, default=10)
    parser.add_argument("--mlp_hidden_size", type=int, default=-1)
    parser.add_argument("--mlp_num_layers", type=int, default=1)
    parser.add_argument("--max_tau", type=float, default=30.)
    
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_prefetch_batch", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--machine", type=str, default="MLcloud")
    parser.add_argument("--save_model", dest="save_model", action="store_true")

    parser.set_defaults(short_run=False, forget_gate=False, learn_mem_tau=False, curriculum=False, Nsum=False, save_model=False)
    
    args = parser.parse_args()

    

    
    ########## Logging Config ##########
    print("Wandb configuration started...")

    # setup directory for saving training artefacts
    temporary_dir = tempfile.TemporaryDirectory()
    artefacts_dir = Path(temporary_dir.name) / "training_artefacts"
    os.makedirs(str(artefacts_dir))

    # wandb config
    api_key_file = Path("~/.wandbAPIkey.txt").expanduser().resolve()
    project_name = "Parity"
    group_name = "N%2d"%(args.N)

    # login to wandb
    with open(api_key_file, "r") as file:
        api_key = file.read().strip()
    wandb.login(key=api_key)

    # initialize wandb
    wandb.init(project=project_name, group=group_name, config={})

    ########## General Config ##########
    print("General configuration started...")

    # General Config
    general_config = dict()
    general_config["seed"] = args.seed
    general_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    general_config["short_training_run"] = args.short_run
    general_config["verbose"] = general_config["short_training_run"]
    torch_device = torch.device(general_config["device"])
    print("Torch Device: ", torch_device)

    # Seeding & Determinism
    os.environ["PYTHONHASHSEED"] = str(general_config["seed"])
    random.seed(general_config["seed"])
    np.random.seed(general_config["seed"])
    torch.manual_seed(general_config["seed"])
    torch.cuda.manual_seed(general_config["seed"])
    torch.backends.cudnn.deterministic = True

    ########## Data, Model and Training Config ##########
    print("Data, model and training configuration started...")

    # NOTE: this step requires you having downloaded the dataset

    # Download Train Data:
    # https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-train-data
    # Download Test Data:
    # https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-test-data # Data_test

    # Location of downloaded folders
    if args.machine=="MLcloud":
        data_dir_path = Path("../../data")#Path("D:/NeuronIO").expanduser().resolve() 
    elif args.machine=="PC":
        data_dir_path = Path("D:/NeuronIO").expanduser().resolve() 
    else:
        print("Unknown location")

    # Data Config

    data_config = dict()
    data_config["N"] = args.N
    data_config['curriculum'] = args.curriculum
    data_config["Nsum"] = args.Nsum


    # Model Config

    model_config = dict()
    model_config["num_input"] = 1
    model_config["num_output"] = 3 if data_config["Nsum"] else 2
    model_config["num_memory"] = args.num_memory
    model_config["mlp_num_layers"] = args.mlp_num_layers
    model_config["mlp_hidden_size"] = args.mlp_hidden_size if args.mlp_hidden_size>0 else 2*model_config["num_memory"]
    model_config["memory_tau_min"] = 1.0
    model_config["memory_tau_max"] = args.max_tau
    model_config["tau_b_value"] = 0.0
    model_config["learn_memory_tau"] = args.learn_mem_tau
    model_config["num_synapse_per_branch"] = 1

    # Training Config

    train_config = dict()
    train_config['forget_gate'] = args.forget_gate
    train_config["num_epochs"] = 1 if general_config["short_training_run"] else 2000
    train_config["learning_rate"] = 5e-4
    train_config["batch_size"] = 32 if general_config["short_training_run"] else 32
    train_config["batches_per_epoch"] = 2000000 if general_config["short_training_run"] else 2000
    train_config["batches_per_epoch"] = int(8/train_config["batch_size"] * train_config["batches_per_epoch"])
    train_config["num_workers"] = args.num_workers # will make run nondeterministic

    # Save Configs

    with open(str(artefacts_dir / "general_config.json"), "w", encoding="utf-8") as f:
        json.dump(general_config, f, ensure_ascii=False, indent=4, sort_keys=True)
    with open(str(artefacts_dir / "data_config.json"), "w", encoding="utf-8") as f:
        json.dump(data_config, f, ensure_ascii=False, indent=4, sort_keys=True)
    with open(str(artefacts_dir / "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=4, sort_keys=True)
    with open(str(artefacts_dir / "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=4, sort_keys=True)
    wandb.config.update(
        {
            "general_config": general_config,
            "data_config": data_config,
            "model_config": model_config,
            "train_config": train_config,
        }
    )

    ########## Data, Model and Training Setup ##########
    print("Data, model and training setup started...")

    #train_config["learning_rate"] = 1e-6

    # Initialize the ELM model
    if train_config['forget_gate']:
        model = ELMf(**model_config).to(torch_device)
    else:
        model = ELM(**model_config).to(torch_device)

    # Initialize the loss function, optimizer, and scheduler
    CELoss = nn.CrossEntropyLoss()
    MSELoss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    scheduler = CosineAnnealingLR(
        optimizer, T_max=train_config["batches_per_epoch"] * train_config["num_epochs"]
    )

    # Visualize ELM model
    print(model)

    ########## TEAINING ##########
    print("Training started...")

    # Initialize the best validation RMSE to a high value
    best_valid_rmse = float("inf")
    best_acc = float("inf")
    best_model_state_dict = model.state_dict().copy()

    # Training loop
    Ns = [args.N]

    check_loss = []
    check_lr = []
    grad_step = []
    for epoch in range(train_config["num_epochs"]):
        # Training
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        pbar = tqdm(
            range(train_config["batches_per_epoch"]),
            total=train_config["batches_per_epoch"],
            disable=not general_config["verbose"],
        )
        for i in pbar:
            sequences, labels = make_batch_Nbit_pair_paritysum(Ns, train_config["batch_size"], duplicate=1, classify_in_time=True, device=torch_device)
            #parity = labels[0]
            parity, Nsum = labels[0]
            # Perform a single training step
            optimizer.zero_grad()
            outputs = model(sequences)[:, Ns[0]-1:].permute(0, 2, 1)
            loss = CELoss(outputs[:,:2], parity)
            if args.Nsum:
                loss += 1./Ns[0]*MSELoss(outputs[:,2], Nsum) 
            loss.backward()
            #grad_step.append(model.w_y.weight.grad.cpu().norm(2)) 
            optimizer.step()
            scheduler.step()

            _, prediction = torch.max(outputs[:,:2], dim=1)
            #print(prediction.shape)
            acc = ((prediction==parity) * 1.).mean()
            # Update running loss
            running_acc += acc.item()
            running_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1} Loss: {running_loss / (i+1):.5f}")


            
           # if acc.item() >0.95:
            #    break

        avg_acc = running_acc / train_config["batches_per_epoch"]
        # Copy model state dict if validation RMSE has improved
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_model_state_dict = model.state_dict().copy()
            

        # Print statistics
        print(
            f"Epoch: {epoch+1}, "
            f'Train Loss: {running_loss / train_config["batches_per_epoch"]:.5f}, '
            f'Train Accuracy: {avg_acc:.5f}'
        )
        
        if avg_acc > 0.95:
            print(
                f"Epoch: {epoch+1}, "
                f'Train Loss: {running_loss / train_config["batches_per_epoch"]:.5f}, '
                f'Train Accuracy: {avg_acc:.5f}'
            )
            print("N=%d solved"%Ns[0])
            if args.curriculum:
                Ns[0] += 1
            else:
                break
            

        # Log statistics
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / train_config["batches_per_epoch"],
                "train_acc": avg_acc,
            }
        )

    #print(grad_step[0].shape)
    #grad_step = torch.concat(grad_step, dim = 0).numpy()
    #print(grad_step.shape)
    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(check_lr, check_loss)
    # axs[0].set_xscale('log')
    # #for i in grad_step:
    # axs[1].plot(grad_step)
    # plt.show()
    print(model.tau_m)
    # Free up memory
    gc.collect()
    # TODO: copy artefacts_dir for local version

    # save model for later
    if args.save_model:
        torch.save(
            best_model_state_dict, str("../models/new_exp/parity_best_model_forget_%r_N_%r_nummem_%d.pt"%(args.forget_gate, Ns[0], args.num_memory))
        )

    # save artefacts to wandb
    wandb.save(
        str(artefacts_dir) + "/*", base_path=str(temporary_dir.name), policy="now"
    )
    wandb.finish()  # finish wandb run
    temporary_dir.cleanup()

    ########## FINISHED ##########
    print("Finished")
