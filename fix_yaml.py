import re
with open('.github/workflows/ci.yml', 'r') as f:
    content = f.read()

content = content.replace('uses: gitleaks/gitleaks-action@v2', 'uses: gitleaks/gitleaks-action@v1.6.0')

with open('.github/workflows/ci.yml', 'w') as f:
    f.write(content)
