#!/usr/bin/env bash
# Validate Kubernetes manifests using kubectl dry-run
#
# Usage: ./scripts/validate_k8s.sh

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "☸️  Validating Kubernetes manifests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check for kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Install kubectl to validate manifests."
    echo "   See: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi

echo "✅ kubectl found"
echo ""

# Validate each manifest with dry-run
MANIFESTS=(
    "k8s/namespace.yaml"
    "k8s/configmap.yaml"
    "k8s/secret.yaml"
    "k8s/pvc.yaml"
    "k8s/deployment.yaml"
    "k8s/service.yaml"
    "k8s/job.yaml"
    "k8s/cronjob.yaml"
    "k8s/gpu-resource-quota.yaml"
)

for manifest in "${MANIFESTS[@]}"; do
    echo "Validating $manifest..."
    if kubectl apply --dry-run=client -f "$manifest" > /dev/null 2>&1; then
        echo "  ✅ Valid"
    else
        echo "  ❌ Invalid"
        kubectl apply --dry-run=client -f "$manifest"
        exit 1
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All Kubernetes manifests are valid"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
