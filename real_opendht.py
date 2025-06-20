import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from kademlia.network import Server
import os
import zlib, base64

logging.basicConfig(level=logging.WARNING)
logging.getLogger("kademlia").setLevel(logging.WARNING)
logging.getLogger("rpcudp").setLevel(logging.WARNING)

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

def _to_serializable(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    else:
        return obj

class RealDhtNode:
    """Kademlia DHT node implementation for Helios."""
    def __init__(self, node_id: str, port: int = 4222, bootstrap_host: str = None, bootstrap_port: int = 4222):
        self.node_id = node_id
        self.port = port
        self.bootstrap_host = bootstrap_host
        self.bootstrap_port = bootstrap_port
        self.is_running = False
        self.peers: Dict[str, PeerInfo] = {}
        self.model_key = "helios_model_params"
        self.server = Server()
        self._bootstrap_nodes = []
        pod_name = os.environ.get("POD_NAME", "helios-node-0")
        print(f"[DHT INIT] POD_NAME={pod_name} HELIOS_BOOTSTRAP_HOST={self.bootstrap_host} PORT={self.port}")
        if pod_name != "helios-node-0":
            print(f"[DHT INIT] Bootstrapping to {self.bootstrap_host}:{self.bootstrap_port}")
            self._bootstrap_nodes.append((self.bootstrap_host, self.bootstrap_port))
        else:
            print("[DHT INIT] This node is the DHT seed (no bootstrap)")

    async def start(self):
        # Determine the external IP address for the node
        pod_name = os.environ.get('HELIOS_POD_NAME', '')
        if pod_name == 'helios-node-0':
            self._bootstrap_nodes = []
        else:
            # All other pods bootstrap to the seed node
            self._bootstrap_nodes = [
                ("helios-node-0.helios-node.helios.svc.cluster.local", 4222)]

        # Get pod ip from downward API
        pod_ip = os.environ.get("POD_IP", "0.0.0.0")
        logger.info(f"[DHT INIT] Using external IP: {pod_ip}")

        self.server = Server()
        await self.server.listen(self.port, interface="0.0.0.0")
        
        if self._bootstrap_nodes:
            while True:
                try:
                    await self.server.bootstrap(self._bootstrap_nodes)
                    logger.info(f"Bootstrapped to {self._bootstrap_nodes}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to bootstrap: {e}, retrying in 5s...")
                    await asyncio.sleep(5)
        
        self.is_running = True
        logger.info(f"Kademlia DHT node started on port {self.port}")

    async def stop(self):
        await self.server.stop()
        self.is_running = False
        logger.info("Kademlia DHT node stopped")
    
    async def put(self, key: str, value: Dict):
        try:
            def force_to_list(obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: force_to_list(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [force_to_list(v) for v in obj]
                else:
                    return obj
            
            serializable_value = force_to_list(value)
            json_str = json.dumps(serializable_value)
            compressed = zlib.compress(json_str.encode('utf-8'))
            b64_encoded = base64.b64encode(compressed).decode('utf-8')
            
            await self.server.set(key, b64_encoded)
            logger.debug(f"Put compressed value to DHT with key: {key}")
            
        except Exception as e:
            logger.error(f"Failed to put value to DHT: {e}")

    async def get(self, key: str) -> Optional[Dict]:
        try:
            result = await self.server.get(key)
            if not result:
                return None

            # Decompress and decode
            compressed = base64.b64decode(result)
            json_str = zlib.decompress(compressed).decode('utf-8')
            data = json.loads(json_str)

            # Track peer if node_id present
            if isinstance(data, dict) and 'node_id' in data and data['node_id'] != self.node_id:
                peer_info = PeerInfo(
                    node_id=data['node_id'],
                    epoch=data.get('epoch', 0),
                    loss=data.get('loss', 0.0),
                    accuracy=data.get('accuracy', 0.0),
                    model_params=data.get('model_params', {}),
                    timestamp=time.time()
                )
                self.peers[peer_info.node_id] = peer_info
            return data
            
        except Exception as e:
            logger.error(f"Failed to get value from DHT: {e}")
            return None
    
    def get_peers(self, exclude_self: bool = True) -> List[PeerInfo]:
        if exclude_self:
            return [peer for peer in self.peers.values() if peer.node_id != self.node_id]
        return list(self.peers.values())
    
    def get_peer_count(self) -> int:
        return len([peer for peer in self.peers.values() if peer.node_id != self.node_id])
    
    async def put_param(self, key: str, value, chunk_size=6000):
        try:
            import torch
            def force_to_list(obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: force_to_list(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [force_to_list(v) for v in obj]
                else:
                    return obj
            value = force_to_list(value)
            json_str = json.dumps(value)
            compressed = zlib.compress(json_str.encode('utf-8'))
            b64 = base64.b64encode(compressed).decode('utf-8')
            # Chunk if needed
            if len(b64) > chunk_size:
                num_chunks = (len(b64) + chunk_size - 1) // chunk_size
                for i in range(num_chunks):
                    chunk = b64[i*chunk_size:(i+1)*chunk_size]
                    chunk_key = f"{key}:chunk{i}"
                    await self.server.set(chunk_key, chunk)
                # Store a meta key with chunk count
                await self.server.set(f"{key}:num_chunks", str(num_chunks))
                logger.debug(f"Put param to DHT with key: {key} (chunked, {num_chunks} chunks)")
            else:
                await self.server.set(key, b64)
                await self.server.set(f"{key}:num_chunks", "1")
                logger.debug(f"Put param to DHT with key: {key} (single chunk)")
        except Exception as e:
            logger.error(f"Failed to put param to DHT: {e}")

    async def get_param(self, key: str):
        try:
            num_chunks_raw = await self.server.get(f"{key}:num_chunks")
            if not num_chunks_raw:
                return None
            num_chunks = int(num_chunks_raw)
            if num_chunks == 1:
                result = await self.server.get(key)
                if not result:
                    return None
                b64 = result
            else:
                chunks = []
                for i in range(num_chunks):
                    chunk = await self.server.get(f"{key}:chunk{i}")
                    if not chunk:
                        return None
                    chunks.append(chunk)
                b64 = ''.join(chunks)
            compressed = base64.b64decode(b64)
            json_str = zlib.decompress(compressed).decode('utf-8')
            data = json.loads(json_str)
            return data
        except Exception as e:
            logger.error(f"Failed to get param from DHT: {e}")
        return None

    async def update_peer_registry(self, node_id: str, epoch: int):
        key = "helios_peer_registry"
        # Fetch current registry
        try:
            registry = await self.get_param(key)
            if not registry:
                registry = {}
            registry[node_id] = epoch
            await self.put_param(key, registry)
            logger.debug(f"Peer registry updated: {registry}")
        except Exception as e:
            logger.error(f"Failed to update peer registry: {e}")

    async def get_peer_registry(self):
        key = "helios_peer_registry"
        try:
            registry = await self.get_param(key)
            if not registry:
                logger.debug("Peer registry fetched: {} (empty)")
                return {}
            logger.debug(f"Peer registry fetched: {registry}")
            return registry
        except Exception as e:
            logger.error(f"Failed to get peer registry: {e}")
            return {}

    async def publish_model_params(self, epoch: int, loss: float, accuracy: float, model_params: dict):
        # Publish each parameter under its own key, chunking if needed
        param_names = list(model_params.keys())
        param_chunks = {}
        for name, value in model_params.items():
            key = f"helios_model_params:{self.node_id}:{name}:{epoch}"
            import torch
            value_list = value
            json_str = json.dumps(value_list)
            compressed = zlib.compress(json_str.encode('utf-8'))
            b64 = base64.b64encode(compressed).decode('utf-8')
            chunk_size = 6000
            if len(b64) > chunk_size:
                num_chunks = (len(b64) + chunk_size - 1) // chunk_size
                param_chunks[name] = num_chunks
            else:
                num_chunks = 1
                param_chunks[name] = 1
            await self.put_param(key, value, chunk_size=chunk_size)
        # Publish metadata
        meta = {
            'node_id': self.node_id,
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy,
            'param_names': param_names,
            'param_chunks': param_chunks,
            'timestamp': time.time()
        }
        meta_key = f"helios_meta:{self.node_id}:{epoch}"
        await self.put_param(meta_key, meta)
        # Update peer registry
        await self.update_peer_registry(self.node_id, epoch)
        logger.debug(f"Published sharded model params for epoch {epoch}")

    async def get_peer_metadata(self, node_id: str, epoch: int):
        key = f"helios_meta:{node_id}:{epoch}"
        return await self.get_param(key)

    async def get_peer_param(self, node_id: str, param_name: str, epoch: int):
        key = f"helios_model_params:{node_id}:{param_name}:{epoch}"
        return await self.get_param(key)

    async def get_latest_peer_epoch(self, node_id: str, max_search: int = 10):
        # Try to find the latest epoch for a peer by searching backwards
        for e in range(0, max_search):
            meta = await self.get_peer_metadata(node_id, e)
            if meta:
                latest = e
        return latest if 'latest' in locals() else None

# Export the main class
dht = RealDhtNode 