# Kubernetes Quick Start Guide

This is a condensed guide to get slower-whisper running on Kubernetes quickly.

## Prerequisites Checklist

- [ ] Kubernetes cluster with GPU nodes
- [ ] NVIDIA GPU Operator or device plugin installed
- [ ] kubectl configured and working
- [ ] Docker installed locally (for building image)
- [ ] Access to a container registry

## 5-Minute Deployment

### 1. Build Container Image

```bash
cd k8s/

# Option A: Use the build script
./build-and-push.sh your-registry.com latest

# Option B: Manual build
docker build -t your-registry.com/slower-whisper:latest -f Dockerfile ..
docker push your-registry.com/slower-whisper:latest
```

### 2. Configure Deployment

Edit these files:

**`deployment.yaml`** - Update line 87:
```yaml
image: your-registry.com/slower-whisper:latest
```

**`deployment.yaml`** - Update line 29 with your GPU node name:
```yaml
nodeSelector:
  kubernetes.io/hostname: your-gpu-node-name  # Find with: kubectl get nodes
```

**`configmap.yaml`** - Optional: change model or language:
```yaml
WHISPER_MODEL: "large-v3"  # or "base", "small", "medium"
WHISPER_LANGUAGE: "en"     # or "auto" for auto-detection
```

### 3. Deploy to Kubernetes

```bash
# Deploy everything at once
kubectl apply -f namespace.yaml
kubectl apply -f pvc.yaml
kubectl apply -f configmap.yaml
kubectl apply -f gpu-resource-quota.yaml
kubectl apply -f deployment.yaml

# Or use Kustomize (one command)
kubectl apply -k .
```

### 4. Verify Deployment

```bash
# Check if pod is running
kubectl get pods -n slower-whisper

# Should show: STATUS = Running
# NAME                              READY   STATUS    RESTARTS   AGE
# slower-whisper-xxxxxxxxxx-xxxxx   1/1     Running   0          2m

# View logs
kubectl logs -n slower-whisper -l app=slower-whisper -f
```

### 5. Process Audio Files

```bash
# Get pod name
POD=$(kubectl get pod -n slower-whisper -l app=slower-whisper -o jsonpath='{.items[0].metadata.name}')

# Upload audio file
kubectl cp your-audio.wav slower-whisper/$POD:/data/raw_audio/

# Run transcription
kubectl exec -n slower-whisper $POD -- \
  uv run slower-whisper transcribe --model large-v3 --device cuda

# Download results
kubectl cp slower-whisper/$POD:/data/whisper_json/your-audio.json ./results/
```

## Common Commands Reference

### Pod Management

```bash
# List pods
kubectl get pods -n slower-whisper

# Describe pod (for troubleshooting)
kubectl describe pod -n slower-whisper -l app=slower-whisper

# Shell into pod
kubectl exec -it -n slower-whisper deployment/slower-whisper -- bash

# View logs (follow)
kubectl logs -n slower-whisper -l app=slower-whisper -f

# Restart deployment
kubectl rollout restart deployment slower-whisper -n slower-whisper
```

### File Operations

```bash
# Upload single file
kubectl cp local-file.wav slower-whisper/$POD:/data/raw_audio/

# Upload directory
kubectl cp ./local-dir/ slower-whisper/$POD:/data/raw_audio/

# Download transcripts
kubectl cp slower-whisper/$POD:/data/whisper_json/ ./results/

# List files in pod
kubectl exec -n slower-whisper $POD -- ls -la /data/raw_audio/
```

### Running Jobs

```bash
# Deploy as one-time job instead of deployment
kubectl apply -f job.yaml

# Monitor job
kubectl get jobs -n slower-whisper -w

# View job logs
kubectl logs -n slower-whisper job/slower-whisper-job -f

# Delete completed job
kubectl delete job slower-whisper-job -n slower-whisper
```

### Configuration Changes

```bash
# Edit ConfigMap
kubectl edit configmap slower-whisper-config -n slower-whisper

# Or apply updated file
kubectl apply -f configmap.yaml

# Restart pods to pick up changes
kubectl rollout restart deployment slower-whisper -n slower-whisper
```

## Deployment Patterns

### Pattern 1: Long-Running Service (Default)

Use `deployment.yaml` for continuous operation:

```bash
kubectl apply -f deployment.yaml
```

**When to use:**
- Continuous monitoring of audio directory
- Running as HTTP API (future enhancement)
- Always-on processing

### Pattern 2: Batch Job

Use `job.yaml` for one-time processing:

```bash
kubectl apply -f job.yaml
```

**When to use:**
- Process a batch of files once
- Scheduled processing (combine with CronJob)
- CI/CD pipeline integration

### Pattern 3: Scheduled Processing

Use `cronjob.yaml` for recurring jobs:

```bash
kubectl apply -f cronjob.yaml
```

**When to use:**
- Process new files every N hours
- Automated nightly transcription
- Regular batch processing

## Troubleshooting Quick Fixes

### Pod won't start (Pending state)

```bash
# Check why
kubectl describe pod -n slower-whisper -l app=slower-whisper

# Common fixes:
# - No GPU available: Check GPU nodes with 'kubectl get nodes'
# - PVC not bound: Check with 'kubectl get pvc -n slower-whisper'
# - Resource quota: Check with 'kubectl describe resourcequota -n slower-whisper'
```

### GPU not detected

```bash
# Verify GPU in pod
kubectl exec -n slower-whisper deployment/slower-whisper -- nvidia-smi

# If fails, check NVIDIA device plugin:
kubectl get daemonset -n kube-system | grep nvidia
```

### Out of memory

```bash
# Quick fix: Use smaller model
kubectl set env deployment/slower-whisper WHISPER_MODEL=base -n slower-whisper

# Or increase memory limit in deployment.yaml:
# memory: "64Gi"  # Increase this
```

### Models won't download

```bash
# Check internet access
kubectl exec -n slower-whisper deployment/slower-whisper -- curl -I https://huggingface.co

# If behind proxy, set HTTP_PROXY in deployment.yaml
```

## Resource Sizing Guide

### Small Jobs (< 1 hour audio)
- Model: `base` or `small`
- GPU: 1x any NVIDIA GPU
- Memory: 8-16 GB
- Storage: 20 GB

### Medium Jobs (1-10 hours audio)
- Model: `medium` or `large-v3`
- GPU: 1x Tesla T4 or better
- Memory: 16-32 GB
- Storage: 50-100 GB

### Large Jobs (> 10 hours audio)
- Model: `large-v3`
- GPU: 1x Tesla V100/A100
- Memory: 32-64 GB
- Storage: 100-500 GB

## Next Steps

1. **Monitor resource usage:**
   ```bash
   kubectl top pod -n slower-whisper
   ```

2. **Set up automated backups:**
   ```bash
   # Add to CronJob to backup transcripts
   kubectl cp slower-whisper/$POD:/data/whisper_json/ ./backups/$(date +%Y%m%d)/
   ```

3. **Scale for multiple GPUs:**
   - Edit `deployment.yaml`: Set `replicas: 2`
   - Ensure you have 2+ GPU nodes available

4. **Add monitoring:**
   - Install Prometheus for metrics
   - Set up Grafana dashboards
   - Configure log aggregation

## Getting Help

- Full documentation: See [README.md](README.md)
- Project documentation: See [../README.md](../README.md)
- Kubernetes docs: https://kubernetes.io/docs/
- NVIDIA GPU Operator: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/

## Cleanup

```bash
# Delete everything
kubectl delete namespace slower-whisper

# Or delete individually
kubectl delete -f deployment.yaml
kubectl delete -f configmap.yaml
kubectl delete -f pvc.yaml  # Warning: deletes all data!
kubectl delete -f namespace.yaml
```
