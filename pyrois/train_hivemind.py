import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from hivemind import DHT, Optimizer
from hivemind.utils import get_logger
import os
import argparse

logger = get_logger(__name__)

# Define a simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 784) # Flatten MNIST images
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_model(args):
    # Setup device
    device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize DHT
    if args.initial_peers:
        dht = DHT(initial_peers=args.initial_peers.split(','), start=True)
    else:
        # For the first peer (remote machine), explicitly bind to 0.0.0.0 for IPv4
        # and advertise the specific public IPv4 address.
        dht = DHT(start=True, host_maddrs=[f"/ip4/0.0.0.0/tcp/{args.dht_port}"], # <<< IPv4 ONLY for listening
                  announce_maddrs=[f"/ip4/{args.public_ip}/tcp/{args.dht_port}"]) # <<< IPv4 ONLY for announcing
    
    # Print the visible multiaddrs for other peers to connect
    print("DHT visible multiaddrs:")
    for addr in dht.get_visible_maddrs():
        print(f"- {addr}")

    # Initialize model and optimizer
    model = SimpleMLP().to(device)
    base_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # --- Hivemind Optimizer setup ---
    opt = Optimizer(
        dht=dht,
        optimizer=base_optimizer,
        run_id=args.run_id,
        target_batch_size=args.target_batch_size,
        batch_size_per_step=args.batch_size,
    )

    logger.info("Attempting to load state from peers...")
    opt.load_state_from_peers()
    logger.info("State loaded or initialized.")

    # Main training loop
    logger.info(f"Starting training for run_id: {args.run_id}")
    for epoch in range(args.num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            opt.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            opt.step()

            if batch_idx % args.log_interval == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)} \tLoss: {loss.item():.6f}')
        
        logger.info(f"Epoch {epoch} finished.")

    logger.info("Training finished!")
    if args.save_model:
        torch.save(model.state_dict(), f"{args.run_id}_final_model.pt")
        logger.info(f"Model saved to {args.run_id}_final_model.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hivemind Distributed Training PoC')
    parser.add_argument('--run_id', type=str, default='mnist_poc_run',
                        help='Unique identifier for this training run')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for local training (default: 32)')
    parser.add_argument('--target_batch_size', type=int, default=128,
                        help='Target global batch size for Hivemind (default: 128). Needs to be >= sum of local batch sizes.')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--cuda_device', type=int, default=0,
                        help='CUDA device index to use (default: 0)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='How many batches to wait before logging training status (default: 10)')
    parser.add_argument('--initial_peers', type=str, default=None,
                        help='Comma-separated list of multiaddresses of initial DHT peers (e.g., "/ip4/X.X.X.X/tcp/Y/p2p/Qm...").')
    parser.add_argument('--dht_port', type=int, default=8080,
                        help='Port for the DHT to listen on (only relevant if initial_peers is not set)')
    parser.add_argument('--public_ip', type=str, default=None,
                        help='Public IP address of this peer for advertising to others. REQUIRED for initial peer.')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the final model state dict')
    parser.add_argument('--client_mode', action='store_true',
                        help='If true, this peer will only connect and not accept incoming connections. Useful if behind strict NAT without IPFS.')
    parser.add_argument('--use_ipfs', action='store_true',
                        help='Use IPFS for NAT traversal (experimental)')
    
    args = parser.parse_args()
    train_model(args)