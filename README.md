# Helios - Distributed AI Training System

A decentralized AI training framework using Kubernetes and OpenDHT for peer-to-peer communication.

## Features

- **Kubernetes Scaling**: Runs with multiple nodes for distributed training
- **Real-time Monitoring**: Live terminal logs with Ctrl+C to stop and cleanup
- **Peer-to-peer Communication**: Uses OpenDHT for distributed parameter sharing
- **Early Stopping**: Intelligent training termination based on convergence criteria
- **Resource Management**: Optimized for limited resources with configurable limits

## Quick Start

### Prerequisites
- Docker Desktop with Kubernetes enabled
- Python 3.8+
- kubectl CLI tool

### Start Training
```bash
python run_helios.py start
```

This single command will:
1. Build the Docker image with OpenDHT
2. Deploy to Kubernetes with auto-scaling
3. Start training with live monitoring
4. Stop and cleanup on Ctrl+C

### Cleanup
```bash
python run_helios.py cleanup
```

## Manual Kubernetes Commands

### Check System Status
```bash
# Check pods
kubectl get pods -n helios

# Check HPA (Horizontal Pod Autoscaler)
kubectl get hpa -n helios

# Check resource usage
kubectl top pods -n helios

# Check events
kubectl get events -n helios --sort-by=.metadata.creationTimestamp
```

### View Logs
```bash
# View logs from all pods
kubectl logs -f -l app=helios-node -n helios

# View logs from specific pod
kubectl logs -f <pod-name> -n helios

# View logs from previous pod (if restarted)
kubectl logs -f <pod-name> -n helios --previous
```

### Scale Manually
```bash
# Scale to specific number of replicas
kubectl scale statefulset helios-node --replicas=3 -n helios

# Scale to 0 to stop training
kubectl scale statefulset helios-node --replicas=0 -n helios
```

### Debug Issues
```bash
# Describe pods for detailed status
kubectl describe pods -n helios

# Check pod resource limits
kubectl describe deployment helios-node -n helios

# Check HPA configuration
kubectl describe hpa helios-node -n helios
```

## System Architecture

### Components
- **Helios Node** (`helios_node.py`): Core training node with PyTorch model
- **Real OpenDHT** (`real_opendht.py`): Peer-to-peer communication layer
- **Mock OpenDHT** (`mock_opendht.py`): Fallback for testing
- **Kubernetes Configs** (`k8s/`): StatefulSet and configuration files
- **Management Script** (`run_helios.py`): Single command to start/stop everything

### Scaling Configuration
- **Default replicas**: 2 nodes
- **Scaling**: Manual scaling via kubectl
- **Node Discovery**: Automatic via StatefulSet DNS names
- **Network**: Headless service for peer communication

## File Structure

```
pyrois/
├── helios_node.py          # Main training node
├── real_opendht.py         # Real OpenDHT implementation
├── mock_opendht.py         # Mock DHT for testing
├── run_helios.py           # Management script
├── Dockerfile              # Container with OpenDHT
├── requirements.txt        # Python dependencies
├── k8s/                    # Kubernetes configurations
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── deployment.yaml
│   └── hpa.yaml
└── README.md               # This documentation
```

## How It Works

1. **Node Discovery**: Each node joins the OpenDHT network and discovers peers
2. **Parameter Sharing**: Nodes periodically share their model parameters via DHT
3. **Federated Averaging**: Received parameters are averaged with local parameters
4. **Auto-scaling**: Kubernetes HPA scales nodes based on resource usage
5. **Early Stopping**: Training stops when convergence criteria are met

## Monitoring

### Live Logs
```bash
kubectl logs -f -l app=helios-node -n helios
```

### System Status
```bash
kubectl get pods -n helios
kubectl get hpa -n helios
kubectl top pods -n helios
```

### Health Checks
Each node exposes health endpoints:
- `/health` - Liveness probe
- `/ready` - Readiness probe  
- `/metrics` - Resource metrics
- `/status` - Training status

## Configuration

Environment variables are set in `k8s/configmap.yaml`:

- `HELIOS_MAX_EPOCHS`: Maximum training epochs (default: 1000)
- `HELIOS_CONVERGENCE_THRESHOLD`: Loss improvement threshold (default: 0.001)
- `HELIOS_PATIENCE`: Early stopping patience (default: 10)
- `HELIOS_MIN_ACCURACY`: Minimum accuracy threshold (default: 95.0)
- `HELIOS_BATCH_SIZE`: Training batch size (default: 32)
- `HELIOS_LEARNING_RATE`: Learning rate (default: 0.001)

## Troubleshooting

### Pods Not Starting
```bash
# Check pod events
kubectl describe pods -n helios

# Check if image exists
docker images | grep helios-node

# Rebuild image if needed
docker build -t helios-node:latest .
```

### High Resource Usage
```bash
# Check resource usage
kubectl top pods -n helios

# Check HPA status
kubectl get hpa -n helios

# Adjust resource limits in k8s/deployment.yaml if needed
```

### Training Not Progressing
```bash
# Check logs for errors
kubectl logs -f -l app=helios-node -n helios

# Check if DHT nodes can communicate
kubectl logs <pod-name> -n helios | grep -i "peer\|dht"
```

### Resource Limits
- **Memory**: 256Mi request, 512Mi limit per node
- **CPU**: 100m request, 250m limit per node
- **Storage**: Minimal (logs only)

## Development

### Adding Features
- Modify `helios_node.py` for training logic changes
- Update `real_opendht.py` for DHT communication changes
- Adjust `k8s/deployment.yaml` for resource changes
- Update `run_helios.py` for management changes

### Testing
The system includes a mock DHT implementation for testing without OpenDHT dependencies.

## License

This project uses OpenDHT which is licensed under GPL-3.0. 