import os

try:
    os.makedirs(".runs/pr_takeover/210_20260126_013000/meta")
    os.makedirs(".runs/pr_takeover/210_20260126_013000/evidence")
    print("Directories created")
except Exception as e:
    print(f"Error: {e}")
