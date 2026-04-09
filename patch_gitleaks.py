with open('.github/workflows/ci.yml', 'r') as f:
    content = f.read()

content = content.replace(
    'GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}',
    'GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}\n          GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}'
)

with open('.github/workflows/ci.yml', 'w') as f:
    f.write(content)
