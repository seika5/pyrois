#!/usr/bin/env python3
"""
Helios Training Management Script
Provides simple commands to start, stop, and monitor distributed training.
"""

import subprocess
import sys
import time
import argparse
import signal
import threading
from typing import List

# Global flag for graceful shutdown
shutdown_requested = False

# Replace all deployment scaling/cleanup with statefulset
START_CMD = ["kubectl", "scale", "statefulset", "helios-node", "--replicas=2", "-n", "helios"]
STOP_CMD = ["kubectl", "scale", "statefulset", "helios-node", "--replicas=0", "-n", "helios"]

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    print("\n🛑 Shutdown requested... Stopping training gracefully...")
    shutdown_requested = True

def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=check)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result

def run_command_live(cmd: List[str]):
    """Run a command and stream output live to terminal."""
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              text=True, bufsize=1, universal_newlines=True, encoding='utf-8', errors='replace')
    
    try:
        for line in iter(process.stdout.readline, ''):
            if shutdown_requested:
                break
            print(line.rstrip())
        process.stdout.close()
        return_code = process.wait()
        if return_code:
            print(f"Command exited with code {return_code}")
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        process.terminate()
        process.wait()

def check_prerequisites():
    """Check if Docker and kubectl are available."""
    try:
        run_command(["docker", "--version"])
        run_command(["kubectl", "version", "--client"])
        print("✓ Prerequisites check passed")
    except subprocess.CalledProcessError:
        print("✗ Error: Docker and kubectl must be installed and available")
        sys.exit(1)

def build_image():
    """Build the Docker image."""
    print("Building Docker image...")
    run_command(["docker", "build", "-t", "helios-node:latest", "."])
    print("✓ Docker image built successfully")

def prune_docker_images():
    """Remove dangling Docker images."""
    print("Pruning dangling Docker images...")
    run_command(["docker", "image", "prune", "-f"], check=False)
    print("✓ Docker images pruned.")

def prune_docker_builder_cache():
    """Remove all Docker builder cache."""
    print("Pruning Docker builder cache...")
    run_command(["docker", "builder", "prune", "-f"], check=False)
    print("✓ Docker builder cache pruned.")

def apply_k8s_configs():
    """Apply Kubernetes configurations."""
    print("Applying Kubernetes configurations...")
    run_command(["kubectl", "apply", "-f", "k8s/namespace.yaml"])
    run_command(["kubectl", "apply", "-f", "k8s/configmap.yaml"])
    run_command(["kubectl", "apply", "-f", "k8s/statefulset-podname-env.yaml"])
    print("✓ Kubernetes configurations applied")

def start_training():
    """Start the training by scaling up the deployment."""
    print("Starting Helios training...")
    run_command(START_CMD)
    print("✓ Training started with 2 nodes")
    print("The system will auto-scale based on resource usage (1-5 nodes)")
    
    # Wait a bit for pods to start
    print("Waiting for pods to start...")
    time.sleep(10)
    
    # Check pod status
    print("Checking pod status...")
    run_command(["kubectl", "get", "pods", "-n", "helios"])
    
    # If pods are still creating, show more details
    result = subprocess.run(["kubectl", "get", "pods", "-n", "helios", "--no-headers"], 
                          capture_output=True, text=True)
    if "ContainerCreating" in result.stdout:
        print("Pods are still creating. Checking for issues...")
        run_command(["kubectl", "describe", "pods", "-n", "helios"])
        run_command(["kubectl", "get", "events", "-n", "helios", "--sort-by=.metadata.creationTimestamp"])

def stop_training():
    """Stop the training by scaling down the deployment."""
    print("Stopping Helios training...")
    try:
        run_command(STOP_CMD, check=False)
        print("✓ Training stopped")
    except subprocess.CalledProcessError:
        print("ℹ️ No deployment to stop (may not exist yet)")

def show_logs():
    """Show logs from all pods."""
    print("=== Helios Node Logs ===")
    run_command_live(["kubectl", "logs", "-f", "-l", "app=helios-node", "-n", "helios"])

def cleanup():
    """Clean up all resources."""
    print("Cleaning up all Helios resources...")
    try:
        run_command(["kubectl", "delete", "namespace", "helios"], check=False)
        print("✓ All resources cleaned up")
    except subprocess.CalledProcessError:
        print("ℹ️ No namespace to clean up (may not exist yet)")

def start_and_monitor(no_build=False):
    """Start training and monitor live with Ctrl+C to stop."""
    global shutdown_requested
    
    # Setup signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print("🚀 Starting Helios Training System...")
        
        # Build and setup
        check_prerequisites()
        if not no_build:
            build_image()
            prune_docker_images()
        else:
            print("Skipping image build as requested.")
        
        apply_k8s_configs()
        
        # Start training
        start_training()
        
        print("\n📊 Training started! Monitoring live logs...")
        print("Press Ctrl+C to stop training and cleanup")
        print("=" * 60)
        
        print("Waiting for pods to start...")
        # Wait for all pods to be ready (up to 2 minutes)
        max_wait = 120
        waited = 0
        while waited < max_wait:
            result = run_command(["kubectl", "get", "pods", "-n", "helios"])
            lines = result.stdout.strip().splitlines()
            if len(lines) > 1:
                statuses = [line.split()[2] for line in lines[1:] if len(line.split()) > 2]
                if all(s == "Running" for s in statuses):
                    break
            time.sleep(3)
            waited += 3
        print("Pods status after wait:")
        run_command(["kubectl", "get", "pods", "-n", "helios"])
        print("\n\U0001F4CA Training started! Monitoring live logs...")
        print("Press Ctrl+C to stop training and cleanup")
        print("=" * 60)
        print("=== Helios Node Logs ===")
        # Wait for at least one pod to be ready for logs
        waited = 0
        while waited < max_wait:
            result = run_command(["kubectl", "get", "pods", "-n", "helios"])
            lines = result.stdout.strip().splitlines()
            if len(lines) > 1:
                ready = any(line.split()[2] == "Running" for line in lines[1:] if len(line.split()) > 2)
                if ready:
                    break
            time.sleep(3)
            waited += 3
        # Now tail logs
        run_command_live(["kubectl", "logs", "-f", "-l", "app=helios-node", "-n", "helios"])
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(e.cmd)}")
        if e.stdout:
            print("--- stdout ---")
            print(e.stdout)
        if e.stderr:
            print("--- stderr ---")
            print(e.stderr)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        print("\n🧹 Cleaning up...")
        stop_training()
        cleanup()
        print("✅ Cleanup complete!")

def main():
    parser = argparse.ArgumentParser(description="Helios Training Management")
    parser.add_argument("command", choices=[
        "start", "cleanup", "docker-prune"
    ], help="Command to execute")
    parser.add_argument("--no-build", action="store_true", help="Skip Docker image build")
    
    args = parser.parse_args()
    
    if args.command == "start":
        start_and_monitor(no_build=args.no_build)
    
    elif args.command == "cleanup":
        cleanup()

    elif args.command == "docker-prune":
        prune_docker_images()
        prune_docker_builder_cache()

if __name__ == "__main__":
    main() 