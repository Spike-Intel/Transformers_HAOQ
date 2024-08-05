import argparse
from pathlib import Path
from dataclasses import dataclass

from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from HAOQ import Config, HAOQLMHeadModel
from HAOQ.utils import tokenize_and_concatenate, get_time

def get_output_dirs(output_dir: Path | str):
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    return output_dir, ckpt_dir, logs_dir

def get_dataloaders(
    dataset_name: str = "NeelNanda/pile-10k",
    tokenizer_name: str = "gpt2",
    max_length: int = 1024,
    batch_size: int = 8,
    column_name: str = "text"
):
    # load dataset
    ds = load_dataset(dataset_name, split='train')

    # process dataset
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    ds = tokenize_and_concatenate(ds, tokenizer, max_length=max_length, column_name=column_name).train_test_split(test_size=100)

    train_loader = torch.utils.data.DataLoader(ds['train'], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(ds['test'], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def step(
    model,
    batch,
    criteria,
    optimizer=None
):

    tokens = batch
    logits = model(tokens)

    loss = criteria(logits.permute(0, 2, 1)[:,:, :-1], tokens[:, 1:])

    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item()

def main(
    model_config: Config,
    dataset_name: str = "NeelNanda/pile-10k",
    n_epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    max_length: int = 1024,
    column_name: str = "text",
    max_steps: int = 1000,
    test_every: int = 10,
    ckpt_every: int = 100,
    output_dir: Path | str = Path.home() / "models",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):

    timestamp = get_time()
    # Setup output paths
    model_dir, ckpt_dir, logs_dir = get_output_dirs(output_dir)

    model_id_str = f"{model_config.d_model}_{model_config.n_heads}_{model_config.n_layers}_{timestamp}"

    logs_filepath_train = logs_dir / f"train_{model_id_str}.csv"
    logs_filepath_test = logs_dir / f"test_{model_id_str}.csv"

    if not logs_filepath_train.exists():
        logs_filepath_train.write_text("timestamp,epoch,step,train_loss\n")

    if not logs_filepath_test.exists():
        logs_filepath_test.write_text("timestamp,epoch,step,train_loss\n")

    # Dataloader
    train_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        max_length=max_length,
        batch_size=batch_size,
        column_name=column_name
    )

    # create model, optimizer, and loss function
    model = HAOQLMHeadModel(model_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criteria = torch.nn.CrossEntropyLoss()

    # hold the minimum loss values for display
    # TODO: early stopping
    min_loss_train = float('inf')
    min_loss_test = float('inf')
    prev_min_loss_test = min_loss_test

    try:
        # training loop
        model.train()
        with open(logs_filepath_train, 'a') as logfile_train, open(logs_filepath_test, 'a') as logfile_test:

            epoch_pbar = tqdm(range(n_epochs), desc="Epoch: 0")

            for epoch in range(n_epochs):
                epoch_pbar.set_description(f"Epoch: {epoch}, Train: {min_loss_train:.4f}, Test:{min_loss_test:.4f})")

                batch_pbar = tqdm(enumerate(train_loader), total=min(max_steps, len(train_loader)), desc="Step: 0, Loss: ()", leave=False)
                for step_num, train_batch in batch_pbar:
                    step_loss = step(
                        model,
                        train_batch['tokens'].to(device),
                        criteria,
                        optimizer
                    )

                    # log training loss
                    logfile_train.write(f"{get_time()},{epoch},{step_num},{step_loss:.5f},\n")

                    min_loss_train = min(step_loss, min_loss_train)
                    # update progress bar
                    batch_pbar.set_description(f"Step: {step_num}, Loss: {step_loss:.4f} Min: (Train: {min_loss_train:.4f}, Test: {min_loss_test:.4f})")

                    # save model checkpoint
                    if test_every > 0 and step_num % test_every == 0 and step_num > 0:
                        # test model on separate set
                        model.eval() # Set model to eval mode so it doesn't update weights

                        test_pbar = tqdm(test_loader, total=len(test_loader), desc="Loss: ()", leave=False)
                        for test_batch in test_loader:
                            test_loss = step(
                                model,
                                test_batch['tokens'].to(device),
                                criteria,
                                optimizer=None, # test
                            )

                            test_pbar.set_description(f"Loss: {test_loss:.4f} Min: (Test: {min_loss_test:.4f})")
                            logfile_test.write(f"{get_time()},{epoch},{step_num},{test_loss:.5f},\n")

                            min_loss_test = min(test_loss, min_loss_test)

                        model.train() # Set model back to train mode

                    if ckpt_every > 0 and step_num % ckpt_every == 0 and step_num > 0 and min_loss_test < prev_min_loss_test:
                        print("Saving checkpoint...")
                        torch.save(model.state_dict(), str(ckpt_dir / f'checkpoint_e{epoch:05d}_{step_num:05d}_{model_id_str}.pth'))
                        prev_min_loss_test = min_loss_test

                    if max_steps > 0 and step_num > max_steps: break

                batch_pbar.close()

    except KeyboardInterrupt:
        print("Training interrupted")
    except Exception as e:
        print("ERROR:", e)
        raise e
    finally:
        print("Saving model...")

    # save final model
    torch.save(model.state_dict(), str(model_dir / f'haoq_{model_id_str}.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with configurable parameters.')

    # Config
    parser.add_argument('--debug', type=bool, default=True, help='Debug mode enabled or disabled')
    parser.add_argument('--ln_eps', type=float, default=1e-5, dest='layer_norm_epsilon', help='Layer normalization epsilon')
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=768, help='Dimension of the model')
    parser.add_argument('--d_mlp', type=int, default=3072, help='Dimension of the MLP')
    parser.add_argument('--local_attn_window', type=int, default=128, help='Local attention window size')
    parser.add_argument('--global_attn_window', type=int, default=512, help='Global attention window size')
    parser.add_argument('--ortho_update_freq', type=int, default=100, help='Frequency of orthogonal updates')
    parser.add_argument('--n_ctx', type=int, default=1024, help='Context length')
    parser.add_argument('--n_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')

    # Dataset
    parser.add_argument('--dataset', type=str, default='NeelNanda/pile-10k', dest='dataset_name', help='Name of the dataset to load for training')
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='Name of the tokenizer to use')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length for tokenization')
    parser.add_argument('--column_name', type=str, default='text', help='Column name from the dataset to process')

    # General
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for logs and model checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device used for training')

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--decay', type=float, default=1e-2, dest='weight_decay', help='Weight decay for training')

    # Training
    parser.add_argument('-e', '--n_epochs', type=int, default=10, dest='n_epochs', help='Number of epochs to train')
    parser.add_argument('-b', '--batch_size', type=int, default=8, dest='batch_size', help='Batch size for training and testing')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum number of steps to train')

    # Periodic Actions
    parser.add_argument('--test_every', type=int, default=10, help='Number of steps between testing')
    parser.add_argument('--ckpt_every', type=int, default=100, help='Number of steps between saving checkpoints')


    args = parser.parse_args()
    config = Config(**vars(args))
    main(config)
