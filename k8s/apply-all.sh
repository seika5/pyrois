#!/bin/bash

# Exit on any error
set -e

echo "Applying Kubernetes configurations..."

# Create namespace first
kubectl apply -f namespace.yaml
echo "✓ Namespace created"

# Apply configmap
kubectl apply -f configmap.yaml
echo "✓ ConfigMap created"

# Apply service and statefulset
kubectl apply -f deployment.yaml
echo "✓ Service and StatefulSet created"

# Apply pod name environment config
kubectl apply -f statefulset-podname-env.yaml
echo "✓ Pod name environment config created"

echo "All configurations applied successfully!"

# Show status
echo -e "\nChecking deployment status..."
kubectl get pods -n helios
kubectl get statefulset -n helios 