# Kubernetes Deployment for slower-whisper

This directory contains Kubernetes manifests for deploying slower-whisper on a Kubernetes cluster with GPU support.

## Overview

slower-whisper is a GPU-intensive audio transcription pipeline that requires:
- NVIDIA GPU for hardware acceleration
- Persistent storage for audio data and model cache
- Configuration management for different processing modes

These manifests provide a production-ready deployment with:
- GPU scheduling and resource management
- Persistent volume claims for data and models
- ConfigMap-based configuration
- Multiple deployment options (Deployment, Job, CronJob)
- Security and resource quotas

## Prerequisites

### 1. Kubernetes Cluster with GPU Support

Your cluster must have:
- Kubernetes 1.20+ (1.25+ recommended for CronJob timezone support)
- Nodes with NVIDIA GPUs
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html) or [nvidia-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) installed

Verify GPU nodes:
```bash
kubectl get nodes -o json | jq '.items[].status.capacity."nvidia.com/gpu"'
```

### 2. Container Image

Build and push the slower-whisper Docker image:

```bash
# Create Dockerfile (if not exists)
cat > Dockerfile <<'EOF'
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN pip3 install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY transcription/ ./transcription/

# Install dependencies
RUN uv sync --extra full

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["uv", "run", "slower-whisper", "--help"]
EOF

# Build and push
docker build -t your-registry/slower-whisper:latest .
docker push your-registry/slower-whisper:latest
```

### 3. Storage

Ensure your cluster has:
- StorageClass with sufficient performance (SSD recommended)
- Provisioner for dynamic volume provisioning

List available storage classes:
```bash
kubectl get storageclass
```

## Quick Start

### 1. Update Configuration

Edit the following files before deployment:

**`deployment.yaml`:**
- Update `image:` with your container registry
- Update `nodeSelector` with your GPU node labels
- Adjust `resources` based on your GPU type

**`pvc.yaml`:**
- Set `storageClassName` if using specific storage class
- Adjust `storage` sizes based on your needs

**`configmap.yaml`:**
- Configure Whisper model size and language
- Enable/disable audio enrichment features

### 2. Deploy to Kubernetes

#### Option A: Deploy using kubectl
```bash
# Create namespace
kubectl apply -f namespace.yaml

# Create persistent volumes
kubectl apply -f pvc.yaml

# Create configuration
kubectl apply -f configmap.yaml

# Create resource quotas
kubectl apply -f gpu-resource-quota.yaml

# Deploy application (choose one):

# As Deployment (long-running)
kubectl apply -f deployment.yaml

# As Job (one-time batch processing)
kubectl apply -f job.yaml

# As CronJob (scheduled processing)
kubectl apply -f cronjob.yaml

# Create service (optional, if adding HTTP API)
kubectl apply -f service.yaml
```

#### Option B: Deploy using Kustomize
```bash
# Update kustomization.yaml with your image
# Then deploy everything:
kubectl apply -k .
```

### 3. Verify Deployment

```bash
# Check pod status
kubectl get pods -n slower-whisper

# Check GPU allocation
kubectl describe pod -n slower-whisper | grep nvidia.com/gpu

# View logs
kubectl logs -n slower-whisper -l app=slower-whisper -f

# Check resource usage
kubectl top pod -n slower-whisper
```

## Deployment Options

### Deployment (Long-Running Service)

Use `deployment.yaml` for continuous processing or when running an HTTP API:

```bash
kubectl apply -f deployment.yaml
```

**Characteristics:**
- Stays running indefinitely
- Automatically restarts on failure
- Suitable for API servers or continuous monitoring

**Usage:**
```bash
# Execute commands in running pod
kubectl exec -it -n slower-whisper deployment/slower-whisper -- bash

# Run transcription
kubectl exec -it -n slower-whisper deployment/slower-whisper -- \
  uv run slower-whisper transcribe --model large-v3 --device cuda
```

### Job (One-Time Batch Processing)

Use `job.yaml` for single batch processing runs:

```bash
kubectl apply -f job.yaml
```

**Characteristics:**
- Runs to completion and exits
- Retries on failure (backoffLimit: 3)
- Cleans up after completion

**Usage:**
```bash
# Monitor job
kubectl get jobs -n slower-whisper -w

# View logs
kubectl logs -n slower-whisper job/slower-whisper-job -f

# Delete completed job
kubectl delete job slower-whisper-job -n slower-whisper
```

### CronJob (Scheduled Processing)

Use `cronjob.yaml` for recurring batch processing:

```bash
kubectl apply -f cronjob.yaml
```

**Characteristics:**
- Runs on schedule (default: every 6 hours)
- Prevents concurrent runs
- Keeps history of recent jobs

**Usage:**
```bash
# View cronjob
kubectl get cronjobs -n slower-whisper

# View job history
kubectl get jobs -n slower-whisper

# Manually trigger job
kubectl create job --from=cronjob/slower-whisper-cron manual-run -n slower-whisper

# Suspend cronjob
kubectl patch cronjob slower-whisper-cron -n slower-whisper -p '{"spec":{"suspend":true}}'
```

## Configuration

### ConfigMap Settings

Edit `configmap.yaml` to configure transcription parameters:

```yaml
# Whisper model configuration
WHISPER_MODEL: "large-v3"        # base, small, medium, large-v3
WHISPER_DEVICE: "cuda"           # cuda or cpu
WHISPER_COMPUTE_TYPE: "float16"  # float16 (GPU) or int8 (CPU)
WHISPER_LANGUAGE: "en"           # Language code or "auto"

# Audio enrichment
ENABLE_AUDIO_ENRICHMENT: "true"  # Enable Stage 2 processing
ENABLE_PROSODY: "true"           # Extract pitch, energy, rate
ENABLE_EMOTION: "true"           # Extract emotional features
```

Apply changes:
```bash
kubectl apply -f configmap.yaml

# Restart pods to pick up new config
kubectl rollout restart deployment slower-whisper -n slower-whisper
```

### Secrets (Optional)

For accessing gated HuggingFace models:

```bash
# Create secret
kubectl create secret generic hf-token \
  --from-literal=token=YOUR_HUGGINGFACE_TOKEN \
  -n slower-whisper

# Mount in deployment (add to deployment.yaml):
# env:
#   - name: HF_TOKEN
#     valueFrom:
#       secretKeyRef:
#         name: hf-token
#         key: token
```

## Volume Management

### Upload Audio Files

```bash
# Get pod name
POD=$(kubectl get pod -n slower-whisper -l app=slower-whisper -o jsonpath='{.items[0].metadata.name}')

# Copy audio files to pod
kubectl cp /local/path/audio.wav slower-whisper/$POD:/data/raw_audio/

# Or use a temporary pod for bulk upload
kubectl run uploader --image=busybox -n slower-whisper --rm -it -- sh
# Then mount the PVC and copy files
```

### Download Transcripts

```bash
# Get pod name
POD=$(kubectl get pod -n slower-whisper -l app=slower-whisper -o jsonpath='{.items[0].metadata.name}')

# Copy transcripts from pod
kubectl cp slower-whisper/$POD:/data/whisper_json/transcript.json /local/path/

# Or copy entire directory
kubectl cp slower-whisper/$POD:/data/transcripts/ /local/path/transcripts/
```

### Persistent Volume Structure

```
/data/
├── raw_audio/         # Original audio files (uploaded)
├── input_audio/       # Normalized 16kHz WAV (generated)
├── transcripts/       # TXT and SRT outputs (generated)
└── whisper_json/      # JSON transcripts (generated)

/cache/huggingface/
└── transformers/      # Downloaded models (cached)
```

## Resource Management

### GPU Resources

Each pod requests 1 GPU by default. Adjust in deployment spec:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1  # Number of GPUs
    cpu: "8"
    memory: "32Gi"
  requests:
    nvidia.com/gpu: 1
    cpu: "4"
    memory: "16Gi"
```

### Resource Quotas

The namespace has resource quotas to prevent over-allocation:

```yaml
# Max 2 GPUs in namespace
requests.nvidia.com/gpu: "2"
limits.nvidia.com/gpu: "2"

# Max 16 CPUs requested
requests.cpu: "16"
requests.memory: "64Gi"
```

Modify in `gpu-resource-quota.yaml` as needed.

### Monitoring Resources

```bash
# View resource usage
kubectl top pod -n slower-whisper

# View GPU allocation
kubectl describe node gpu-node | grep nvidia.com/gpu

# View resource quotas
kubectl describe resourcequota -n slower-whisper
```

## GPU Node Selection

### Node Selector (Basic)

Update `deployment.yaml` to target specific GPU nodes:

```yaml
nodeSelector:
  kubernetes.io/hostname: gpu-node-1
  # Or by GPU type:
  accelerator: nvidia-gpu
  gpu-type: nvidia-tesla-v100
```

### Node Affinity (Advanced)

For more complex scheduling:

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: gpu
              operator: In
              values:
                - "true"
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        preference:
          matchExpressions:
            - key: gpu-type
              operator: In
              values:
                - nvidia-tesla-v100
```

### Tolerations

If GPU nodes have taints:

```yaml
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

## Troubleshooting

### Pod Not Scheduling

```bash
# Check events
kubectl describe pod -n slower-whisper

# Common issues:
# - No GPU nodes available
# - Resource quota exceeded
# - Node selector doesn't match any nodes
```

### GPU Not Detected

```bash
# Check if GPU is available
kubectl exec -it -n slower-whisper deployment/slower-whisper -- nvidia-smi

# Check environment variables
kubectl exec -it -n slower-whisper deployment/slower-whisper -- env | grep CUDA

# Verify GPU device plugin
kubectl get daemonset -n kube-system | grep nvidia
```

### Out of Memory

```bash
# Check memory usage
kubectl top pod -n slower-whisper

# Solutions:
# 1. Increase memory limits in deployment.yaml
# 2. Use smaller Whisper model (base or small)
# 3. Reduce batch size in configmap.yaml
```

### Model Download Failures

```bash
# Check internet access from pod
kubectl exec -it -n slower-whisper deployment/slower-whisper -- curl -I https://huggingface.co

# Check disk space
kubectl exec -it -n slower-whisper deployment/slower-whisper -- df -h /cache

# Manually download models (if behind proxy)
kubectl exec -it -n slower-whisper deployment/slower-whisper -- bash
# Inside pod:
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim')"
```

### Persistent Volume Issues

```bash
# Check PVC status
kubectl get pvc -n slower-whisper

# Check PV binding
kubectl get pv | grep slower-whisper

# View PVC events
kubectl describe pvc slower-whisper-data -n slower-whisper
```

## Production Considerations

### High Availability

For critical workloads:

```yaml
# Add pod disruption budget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: slower-whisper-pdb
  namespace: slower-whisper
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: slower-whisper
```

### Monitoring and Logging

```bash
# Enable kubectl logs with labels
kubectl logs -n slower-whisper -l app=slower-whisper --tail=100 -f

# For production, integrate with:
# - Prometheus for metrics
# - Grafana for dashboards
# - ELK/Loki for log aggregation
```

### Backup Strategy

```bash
# Backup transcripts regularly
kubectl exec -n slower-whisper deployment/slower-whisper -- \
  tar czf /tmp/transcripts-backup.tar.gz /data/transcripts /data/whisper_json

kubectl cp slower-whisper/$POD:/tmp/transcripts-backup.tar.gz ./backup.tar.gz
```

### Security Hardening

```yaml
# Add security context
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  capabilities:
    drop:
      - ALL
  readOnlyRootFilesystem: false  # Set to true if possible
```

### Network Policies

```yaml
# Restrict network access
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: slower-whisper-netpol
  namespace: slower-whisper
spec:
  podSelector:
    matchLabels:
      app: slower-whisper
  policyTypes:
    - Ingress
    - Egress
  egress:
    # Allow DNS
    - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
      ports:
        - protocol: UDP
          port: 53
    # Allow HuggingFace
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: TCP
          port: 443
```

## Scaling

### Horizontal Scaling (Multiple GPUs)

If you have multiple GPU nodes:

```yaml
# In deployment.yaml
spec:
  replicas: 2  # Run on 2 GPUs

# Or use Horizontal Pod Autoscaler (requires custom metrics)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: slower-whisper-hpa
  namespace: slower-whisper
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: slower-whisper
  minReplicas: 1
  maxReplicas: 4
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 80
```

### Vertical Scaling (Larger GPU)

For faster processing, use nodes with more powerful GPUs and increase resource limits.

## Cleanup

```bash
# Delete all resources
kubectl delete -f deployment.yaml
kubectl delete -f service.yaml
kubectl delete -f configmap.yaml
kubectl delete -f gpu-resource-quota.yaml
kubectl delete -f pvc.yaml

# Or using Kustomize
kubectl delete -k .

# Delete namespace (removes everything)
kubectl delete namespace slower-whisper
```

## Support

For issues specific to:
- **Kubernetes deployment**: Check this README and cluster logs
- **slower-whisper pipeline**: See main project [README.md](../README.md)
- **GPU scheduling**: Consult [NVIDIA GPU Operator docs](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)

## Additional Resources

- [Kubernetes GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html)
- [Kustomize Documentation](https://kubectl.docs.kubernetes.io/references/kustomize/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
