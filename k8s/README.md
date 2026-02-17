# Kubernetes Manifests

This directory contains Kubernetes deployment assets for running `slower-whisper` with persistent data and optional GPU acceleration.

For a command-by-command guide, see [QUICK_START.md](QUICK_START.md).

## Directory Map

| File | Purpose |
|------|---------|
| `namespace.yaml` | Namespace definition (`slower-whisper`) |
| `configmap.yaml` | Runtime env/config defaults |
| `secret.yaml` | Secret placeholders (HF token/API credentials) |
| `pvc.yaml` | Persistent volume claims for audio/transcript/model data |
| `deployment.yaml` | Long-running workload pattern |
| `service.yaml` | Cluster service exposure |
| `job.yaml` | One-shot batch processing pattern |
| `cronjob.yaml` | Scheduled processing pattern |
| `gpu-resource-quota.yaml` | Optional namespace GPU/resource quota |
| `kustomization.yaml` | Apply bundle via kustomize |
| `Dockerfile` | Container build file used for K8s workflows |
| `build-and-push.sh` | Image build/push helper |
| `docker-compose.yaml` | Local compose helper for K8s-adjacent testing |

## Recommended Deployment Flow

1. Build and push image:

```bash
cd k8s
./build-and-push.sh <registry> <tag>
```

2. Update image reference and any node/storage settings in:

- `k8s/deployment.yaml`
- `k8s/job.yaml`
- `k8s/cronjob.yaml`
- `k8s/pvc.yaml`

3. Apply manifests:

```bash
kubectl apply -k k8s
```

4. Verify workload and logs:

```bash
kubectl get pods -n slower-whisper
kubectl logs -n slower-whisper -l app=slower-whisper -f
```

## Runtime Notes

- GPU usage requires NVIDIA device plugin/operator on the cluster.
- The manifests are designed for local-first processing: audio in PVC, transcript output in PVC.
- Set HF/API secrets in `secret.yaml` (or your secret manager integration) before enabling features that require gated models/providers.
- Choose the right workload pattern:
  - `deployment.yaml` for continuously available processing/service mode.
  - `job.yaml` for one-time batch runs.
  - `cronjob.yaml` for scheduled batch runs.

## Validation

Validate manifests from repo root:

```bash
./scripts/validate_k8s.sh
```

For an executable walkthrough with copy/paste commands, use [k8s/QUICK_START.md](QUICK_START.md).

## Related Docs

- [k8s/QUICK_START.md](QUICK_START.md)
- [docs/DOCKER_DEPLOYMENT_GUIDE.md](../docs/DOCKER_DEPLOYMENT_GUIDE.md)
- [docs/GPU_SETUP.md](../docs/GPU_SETUP.md)
