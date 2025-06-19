#!/usr/bin/env python3
"""
Mock OpenDHT implementation for testing Helios distributed training.
Simulates peer discovery between nodes in the same Kubernetes cluster.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PeerInfo:
    """Information about a discovered peer."""
    node_id: str
    epoch: int
    loss: float
    accuracy: float
    model_params: Dict
    timestamp: float

class MockDhtNode:
    """Mock DHT node that simulates peer discovery for testing."""
    
    def __init__(self, node_id: str, port: int = 4222, bootstrap_host: str = "bootstrap.jami.net", bootstrap_port: int = 4222):
        self.node_id = node_id
        self.port = port
        self.bootstrap_host = bootstrap_host
        self.bootstrap_port = bootstrap_port
        self.is_running = False
        self.peers: Dict[str, PeerInfo] = {}
        self.model_key = "helios_model_params"
        self.mock_discovery_task = None
        
        # Simulate shared storage for peer discovery
        self._shared_peers = {}
        
        logger.info(f"Mock DHT node initialized for {node_id}")
    
    def start(self):
        """Start the mock DHT node."""
        self.is_running = True
        logger.info(f"Mock DHT node started on port {self.port}")
        
        # Start mock discovery loop
        self.mock_discovery_task = asyncio.create_task(self._mock_discovery_loop())
    
    def stop(self):
        """Stop the mock DHT node."""
        self.is_running = False
        if self.mock_discovery_task:
            self.mock_discovery_task.cancel()
        logger.info("Mock DHT node stopped")
    
    def put(self, key: str, value: Dict):
        """Put a value into the mock DHT."""
        # Store in shared storage to simulate DHT
        self._shared_peers[key] = value
        logger.debug(f"Mock: Put value with key: {key}")
    
    def get(self, key: str) -> Optional[Dict]:
        """Get a value from the mock DHT."""
        return self._shared_peers.get(key)
    
    async def _mock_discovery_loop(self):
        """Mock discovery loop that simulates finding other nodes."""
        logger.info("Mock peer discovery loop started")
        
        while self.is_running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Look for other nodes in shared storage
                for key, value in self._shared_peers.items():
                    if key == self.model_key and value.get('node_id') != self.node_id:
                        # Found another node's parameters
                        peer_info = PeerInfo(
                            node_id=value['node_id'],
                            epoch=value['epoch'],
                            loss=value['loss'],
                            accuracy=value['accuracy'],
                            model_params=value['model_params'],
                            timestamp=value['timestamp']
                        )
            
                        # Only add if not already present or if newer
                        if (peer_info.node_id not in self.peers or 
                            peer_info.timestamp > self.peers[peer_info.node_id].timestamp):
                            self.peers[peer_info.node_id] = peer_info
                            logger.info(f"Discovered peer {peer_info.node_id} at epoch {peer_info.epoch}")
                
                # Clean up old peers (older than 60 seconds)
                current_time = time.time()
                old_peers = [node_id for node_id, peer in self.peers.items() 
                           if current_time - peer.timestamp > 60]
                for node_id in old_peers:
                    del self.peers[node_id]
                    logger.debug(f"Removed old peer {node_id}")
                
            except asyncio.CancelledError:
                break
                    except Exception as e:
                logger.error(f"Error in mock discovery loop: {e}")
                await asyncio.sleep(1)
    
    def get_peers(self, exclude_self: bool = True) -> List[PeerInfo]:
        """Get list of discovered peers."""
        if exclude_self:
            return [peer for peer in self.peers.values() if peer.node_id != self.node_id]
        return list(self.peers.values())
    
    def get_peer_count(self) -> int:
        """Get number of discovered peers."""
        return len([peer for peer in self.peers.values() if peer.node_id != self.node_id])
    
    def publish_model_params(self, epoch: int, loss: float, accuracy: float, model_params: Dict):
        """Publish model parameters to the mock DHT."""
        value = {
            'node_id': self.node_id,
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy,
            'model_params': model_params,
            'timestamp': time.time()
        }
        self.put(self.model_key, value)
        logger.debug(f"Published model params for epoch {epoch}")

# Export the main class
dht = MockDhtNode

# Create the mock module
class MockOpenDHT:
    """Mock OpenDHT module."""
    
    DhtRunner = MockDhtNode
    crypto = MockCrypto()
    Value = Value

# Export the mock module
__all__ = ['dht'] 