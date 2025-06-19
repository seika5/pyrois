import asyncio
import json
import os
import pickle
import time
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import psutil
import sys
    from real_opendht import RealDhtNode, PeerInfo
import logging
import hashlib

# Configure logging
logger.remove()
try:
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
logger.add(
    "logs/helios_node_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
)
except Exception as e:
    logger.warning(f"Could not setup file logging: {e}")
    # Fallback to console only logging

logger.add(lambda msg: print(msg, end=""), level="WARNING")

console = Console()

logging.basicConfig(level=logging.DEBUG)

@dataclass
class NodeConfig:
    """Configuration for a Helios node."""
    node_id: str
    port: int = 4222
    http_port: int = 8000  # HTTP port for health server
    bootstrap_host: str = "bootstrap.jami.net"
    bootstrap_port: int = 4222
    model_key: str = "helios_model_params"
    gossip_interval: float = 5.0  # seconds
    training_interval: float = 1.0  # seconds
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs_per_round: int = 1
    max_epochs: int = 10  # Set to 10 for quick runs
    convergence_threshold: float = 0.001  # NEW: Stop when loss improvement < threshold
    patience: int = 10  # NEW: Stop if no improvement for N epochs
    min_accuracy: float = 95.0  # NEW: Stop if accuracy reaches threshold

class SimpleModel(nn.Module):
    """A simple neural network for testing distributed training."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 128, output_size: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def _to_serializable(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    else:
        return obj

class HeliosNode:
    """A Helios training node that participates in distributed training using OpenDHT."""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.node_id = config.node_id
        self.dht_node = None
        self.model = SimpleModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.total_rounds = 0
        self.peer_count = 0
        self.last_gossip_time = 0
        self.training_stats = {
            "loss": [],
            "accuracy": [],
            "peer_count": [],
            "rounds": []
        }
        
        # Early stopping state
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.should_stop = False
        
        # Health server
        self.app = FastAPI(title=f"Helios Node {self.node_id}")
        self._setup_health_endpoints()
        
        logger.info(f"Initialized Helios node {self.node_id} on device {self.device}")
        
        self.num_shards = 2  # For now, hardcode to 2 nodes; can be dynamic
        self.shard_index = self._assign_shard_index()
    
    def _setup_health_endpoints(self):
        """Setup health check endpoints."""
        
        @self.app.get("/health")
        async def health_check():
            """Liveness probe endpoint."""
            try:
                cuda_available = torch.cuda.is_available()
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                if cpu_percent > 95 or memory_percent > 95:
                    raise HTTPException(status_code=503, detail="System resources exhausted")
                
                return JSONResponse({
                    "status": "healthy",
                    "node_id": self.node_id,
                    "cuda_available": cuda_available,
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "current_epoch": self.current_epoch,
                    "total_rounds": self.total_rounds,
                    "peer_count": self.peer_count
                })
            except Exception as e:
                raise HTTPException(status_code=503, detail=str(e))
        
        @self.app.get("/ready")
        async def ready_check():
            """Readiness probe endpoint."""
            try:
                return JSONResponse({
                    "status": "ready",
                    "node_id": self.node_id,
                    "dht_connected": self.dht_node is not None
                })
            except Exception as e:
                raise HTTPException(status_code=503, detail=str(e))
        
        @self.app.get("/metrics")
        async def metrics():
            """Metrics endpoint for monitoring."""
            try:
                return JSONResponse({
                    "node_id": self.node_id,
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage('/').percent,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                    "cuda_memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
                    "current_epoch": self.current_epoch,
                    "total_rounds": self.total_rounds,
                    "peer_count": self.peer_count,
                    "latest_loss": self.training_stats["loss"][-1] if self.training_stats["loss"] else 0,
                    "latest_accuracy": self.training_stats["accuracy"][-1] if self.training_stats["accuracy"] else 0
                })
            except Exception as e:
                raise HTTPException(status_code=503, detail=str(e))
        
        @self.app.get("/status")
        async def status():
            """Detailed status endpoint."""
            try:
                return JSONResponse({
                    "node_id": self.node_id,
                    "device": str(self.device),
                    "model_parameters": sum(p.numel() for p in self.model.parameters()),
                    "training_stats": self.training_stats,
                    "config": asdict(self.config)
                })
            except Exception as e:
                raise HTTPException(status_code=503, detail=str(e))
        
    async def start(self):
        """Start the Helios node."""
        try:
            # Start DHT node
            logger.info(f"Starting DHT node on port {self.config.port}")
            self.dht_node = RealDhtNode(self.node_id, self.config.port, self.config.bootstrap_host, self.config.bootstrap_port)
            await self.dht_node.start()
            logger.info(f"Bootstrapping to {self.config.bootstrap_host}:{self.config.bootstrap_port}")
            
            # Start health server
            config = uvicorn.Config(self.app, host="0.0.0.0", port=self.config.http_port, log_level="info")
            server = uvicorn.Server(config)
            self.health_server_task = asyncio.create_task(server.serve())
            
            # Start training and monitoring loops
            self.training_task = asyncio.create_task(self._training_loop())
            self.gossip_task = asyncio.create_task(self._gossip_loop())
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info(f"Helios node {self.node_id} started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Helios node: {e}")
            raise
    
    async def _gossip_loop(self):
        """Periodically share model parameters with other nodes using sharded DHT keys."""
        while True:
            try:
                await asyncio.sleep(self.config.gossip_interval)
                if not self.dht_node or not self.dht_node.is_running:
                    continue
                model_params = self._get_model_state()
                current_loss = self.training_stats["loss"][-1] if self.training_stats["loss"] else 0.0
                current_accuracy = self.training_stats["accuracy"][-1] if self.training_stats["accuracy"] else 0.0
                await self.dht_node.publish_model_params(
                    epoch=self.current_epoch,
                    loss=current_loss,
                    accuracy=current_accuracy,
                    model_params=model_params
                )
                logger.debug(f"Published sharded parameters for epoch {self.current_epoch}")
            except Exception as e:
                logger.error(f"Error in gossip loop: {e}")
                await asyncio.sleep(1)
    
    async def _training_loop(self):
        """Main training loop that also receives and aggregates parameters from peers."""
        last_epoch = -1
        repeat_count = 0
        while True:
            try:
                await asyncio.sleep(self.config.training_interval)
                if self.current_epoch >= self.config.max_epochs:
                    logger.info(f"Reached maximum epochs ({self.config.max_epochs}), stopping training")
                    break
                loss, accuracy = await self._train_epoch()
                self.training_stats["loss"].append(loss)
                self.training_stats["accuracy"].append(accuracy)
                self.training_stats["peer_count"].append(self.peer_count)
                self.training_stats["rounds"].append(self.total_rounds)
                if self.current_epoch == last_epoch:
                    repeat_count += 1
                    if repeat_count > 1:
                        logger.warning(f"Epoch {self.current_epoch} is repeating {repeat_count} times!")
                else:
                    repeat_count = 0
                last_epoch = self.current_epoch
                self.current_epoch += 1
                self.total_rounds += 1
                if self._should_stop_early(loss, accuracy):
                    logger.info(f"Early stopping triggered at epoch {self.current_epoch}")
                    break
                await self._aggregate_peer_parameters()
                logger.info(f"Training round {self.total_rounds}: loss={loss:.4f}, accuracy={accuracy:.4f}, peers={self.peer_count}")
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
        logger.info(f"Training completed after {self.current_epoch} epochs")
        if self.training_stats["loss"]:
            final_loss = self.training_stats["loss"][-1]
            final_accuracy = self.training_stats["accuracy"][-1]
            logger.info(f"Final loss: {final_loss:.4f}, Final accuracy: {final_accuracy:.2f}%")
    
    async def _train_epoch(self) -> Tuple[float, float]:
        """Train the model for one epoch on this node's data shard only."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx in range(10):
            # Generate synthetic data for this shard only
            data_seed = self.shard_index * 1000 + batch_idx
            torch.manual_seed(data_seed)
            data = torch.randn(self.config.batch_size, 1, 28, 28).to(self.device)
            target = torch.randint(0, 10, (self.config.batch_size,)).to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / 10
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    async def _aggregate_peer_parameters(self):
        """Retrieve and aggregate sharded parameters from peer nodes using the DHT peer registry."""
        try:
            if not self.dht_node or not self.dht_node.is_running:
                return
            # Use the DHT peer registry for peer discovery
            registry = await self.dht_node.get_peer_registry()
            logger.debug(f"Peer registry before aggregation: {registry}")
            peer_ids = [nid for nid in registry.keys() if nid != self.node_id]
            if not peer_ids:
                return
            peer_params = []
            self.peer_count = len(peer_ids)
            for peer_id in peer_ids:
                try:
                    latest_epoch = registry[peer_id]
                    meta = await self.dht_node.get_peer_metadata(peer_id, latest_epoch)
                    if not meta or 'param_names' not in meta:
                        continue
                    shard_param_names = [name for idx, name in enumerate(meta['param_names']) if idx % self.num_shards == self.shard_index]
                    peer_shard_params = {}
                    for name in shard_param_names:
                        value = await self.dht_node.get_peer_param(peer_id, name, latest_epoch)
                        if value is not None:
                            peer_shard_params[name] = value
                    if peer_shard_params:
                        peer_params.append(peer_shard_params)
                        logger.debug(f"Fetched sharded params from peer {peer_id} at epoch {latest_epoch}")
                except Exception as e:
                    logger.warning(f"Failed to fetch sharded peer parameters: {e}")
                    continue
            if peer_params:
                await self._federated_averaging(peer_params)
                logger.info(f"Aggregated sharded parameters from {self.peer_count} peers")
        except Exception as e:
            logger.error(f"Error aggregating sharded peer parameters: {e}")
    
    async def _federated_averaging(self, peer_params: List[Dict]):
        """Perform federated averaging of only this node's shard of model parameters."""
        try:
            current_params = self._get_model_state()
            averaged_params = {}
            for key in current_params.keys():
                param_sum = torch.tensor(current_params[key], dtype=torch.float32)
                for peer_param in peer_params:
                    if key in peer_param:
                        param_sum += torch.tensor(peer_param[key], dtype=torch.float32)
                averaged_params[key] = (param_sum / (len(peer_params) + 1)).tolist()
            self._set_model_state(averaged_params)
        except Exception as e:
            logger.error(f"Error in federated averaging: {e}")
    
    def _get_model_state(self) -> Dict:
        """Get only this node's shard of model parameters as a dictionary of lists for JSON serialization."""
        # Shard by splitting named_parameters by index
        params = list(self.model.named_parameters())
        shard_params = {}
        for idx, (name, param) in enumerate(params):
            if idx % self.num_shards == self.shard_index:
                shard_params[name] = _to_serializable(param.clone().detach().cpu())
        return shard_params
    
    def _set_model_state(self, state_dict: Dict):
        """Set only this node's shard of model parameters from a dictionary of lists."""
        with torch.no_grad():
            for idx, (name, param) in enumerate(self.model.named_parameters()):
                if idx % self.num_shards == self.shard_index and name in state_dict:
                    v = state_dict[name]
                    if hasattr(v, 'tolist'):
                        v = v.tolist()
                    param.copy_(torch.tensor(v, dtype=param.dtype, device=param.device))
    
    async def _monitoring_loop(self):
        """Monitor and log system statistics."""
        last_epoch = -1
        last_peer_count = -1
        while True:
            try:
                await asyncio.sleep(10)  # Every 10 seconds
                if self.dht_node and self.dht_node.is_running:
                    self.peer_count = self.dht_node.get_peer_count()
                # Only print if epoch or peer count changed
                if self.current_epoch != last_epoch or self.peer_count != last_peer_count:
                table = Table(title=f"Helios Node {self.node_id} Status")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                table.add_row("Device", str(self.device))
                table.add_row("Current Epoch", str(self.current_epoch))
                table.add_row("Total Rounds", str(self.total_rounds))
                table.add_row("Active Peers", str(self.peer_count))
                table.add_row("Model Parameters", f"{sum(p.numel() for p in self.model.parameters()):,}")
                if self.training_stats["loss"]:
                    table.add_row("Latest Loss", f"{self.training_stats['loss'][-1]:.4f}")
                    table.add_row("Latest Accuracy", f"{self.training_stats['accuracy'][-1]:.2f}%")
                console.print(table)
                    last_epoch = self.current_epoch
                    last_peer_count = self.peer_count
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    async def stop(self):
        """Stop the Helios node."""
        if self.dht_node:
            await self.dht_node.stop()
        logger.info(f"Helios node {self.node_id} stopped")

    def _should_stop_early(self, loss: float, accuracy: float) -> bool:
        """Check if training should stop early based on convergence criteria."""
        # Check if accuracy reached minimum threshold
        if accuracy >= self.config.min_accuracy:
            logger.info(f"Accuracy threshold reached: {accuracy:.2f}% >= {self.config.min_accuracy}%")
            return True
        
        # Check for loss improvement
        loss_improvement = self.best_loss - loss
        if loss_improvement > self.config.convergence_threshold:
            # Good improvement, reset patience counter
            self.best_loss = loss
            self.patience_counter = 0
            logger.debug(f"Loss improved by {loss_improvement:.6f}, resetting patience")
        else:
            # No significant improvement
            self.patience_counter += 1
            logger.debug(f"No significant loss improvement, patience: {self.patience_counter}/{self.config.patience}")
        
        # Check if we've exceeded patience
        if self.patience_counter >= self.config.patience:
            logger.info(f"Patience exceeded: no improvement for {self.config.patience} epochs")
            return True
        
        # Check if accuracy improved
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            logger.debug(f"New best accuracy: {accuracy:.2f}%")
        
        return False

    def _on_peer_discovered(self, peer_info: PeerInfo):
        """Callback when a new peer is discovered."""
        logger.info(f"Discovered peer {peer_info.node_id} at epoch {peer_info.epoch}")
        self.peer_count = self.dht_node.get_peer_count()
    
    def _on_parameters_received(self, peer_info: PeerInfo):
        """Callback when parameters are received from a peer."""
        logger.info(f"Received parameters from peer {peer_info.node_id} (loss: {peer_info.loss:.4f}, accuracy: {peer_info.accuracy:.2f}%)")
        # Ensure model_params are lists before federated averaging
        serializable_params = _to_serializable(peer_info.model_params)
        # Trigger federated averaging with the new parameters
        asyncio.create_task(self._federated_averaging([serializable_params]))

    def _assign_shard_index(self):
        # Use node_id hash to assign a shard index
        return int(hashlib.sha256(self.node_id.encode()).hexdigest(), 16) % self.num_shards

async def main():
    """Main entry point for the Helios node."""
    try:
        import os, time
        pod_name = os.environ.get("POD_NAME", "helios-node-0")
        if pod_name != "helios-node-0":
            print("Delaying startup to allow seed node to initialize...")
            time.sleep(10)
    # Generate unique node ID
    node_id = str(uuid.uuid4())[:8]
    
    # Get configuration from environment variables
    config = NodeConfig(
        node_id=node_id,
        port=int(os.getenv("HELIOS_PORT", "4222")),
        http_port=int(os.getenv("HELIOS_HTTP_PORT", "8000")),
        bootstrap_host=os.getenv("HELIOS_BOOTSTRAP_HOST", "bootstrap.jami.net"),
        bootstrap_port=int(os.getenv("HELIOS_BOOTSTRAP_PORT", "4222")),
        gossip_interval=float(os.getenv("HELIOS_GOSSIP_INTERVAL", "5.0")),
        training_interval=float(os.getenv("HELIOS_TRAINING_INTERVAL", "1.0")),
        batch_size=int(os.getenv("HELIOS_BATCH_SIZE", "32")),
            learning_rate=float(os.getenv("HELIOS_LEARNING_RATE", "0.001")),
            max_epochs=int(os.getenv("HELIOS_MAX_EPOCHS", "10")),
            convergence_threshold=float(os.getenv("HELIOS_CONVERGENCE_THRESHOLD", "0.001")),
            patience=int(os.getenv("HELIOS_PATIENCE", "10")),
            min_accuracy=float(os.getenv("HELIOS_MIN_ACCURACY", "95.0"))
    )
    
    # Create and start node
    node = HeliosNode(config)
    
    try:
        await node.start()
        
            # Keep the node running until training completes
            while node.current_epoch < config.max_epochs:
            await asyncio.sleep(1)
            
            logger.info("Training completed, shutting down node")
                
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await node.stop()
            
    except Exception as e:
        logger.error(f"Fatal error during startup: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 