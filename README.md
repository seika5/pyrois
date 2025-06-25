# Hivemind Kubernetes Deployment

Minimal setup for running hivemind bootstrap and peer nodes on Kubernetes.

## Quick Commands

### Build Images
```bash
# Build both images
docker build -f Dockerfile.bootstrap -t hivemind-bootstrap:latest .
docker build -f Dockerfile.peer -t hivemind-peer:latest .
```

### Run Bootstrap and Peer

1. **Start bootstrap:**
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/bootstrap-deployment.yaml
```

2. **Wait for bootstrap address:**
```bash
kubectl logs deployment/bootstrap-deployment -n hivemind
```
Look for: `To join the training, use initial_peers = ['/ip4/10.1.0.xxx/tcp/xxxxx/p2p/...']`

3. **Update peer with bootstrap address:**
   - Edit `k8s/peer-deployment.yaml` 
   - Replace the `bootstrap_address` value with the address from step 2
   - Make sure to keep the single quotes around the address

4. **Start peer:**
```bash
kubectl apply -f k8s/peer-deployment.yaml
```

**Note:** You need to manually update the bootstrap address in the peer deployment YAML file. It doesn't auto-configure.

### Stop All Nodes
```bash
kubectl delete -f k8s/peer-deployment.yaml
kubectl delete -f k8s/bootstrap-deployment.yaml
kubectl delete -f k8s/namespace.yaml
```

### Cleanup All Images
```bash
docker rmi hivemind-bootstrap:latest
docker rmi hivemind-peer:latest
```

### Monitor Training
```bash
# Bootstrap logs
kubectl logs deployment/bootstrap-deployment -n hivemind -f

# Peer logs  
kubectl logs deployment/peer-deployment -n hivemind -f
```

## Troubleshooting

- **Images not found**: Rebuild with the docker build commands above
- **Connection issues**: Check that bootstrap address in peer deployment matches the actual bootstrap address
- **Pod crashes**: Check logs with `kubectl logs pod/[pod-name] -n hivemind` 