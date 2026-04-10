# Resolving PR merge conflicts for this branch

If GitHub shows merge conflicts for:
- `scripts/infer.py`
- `scripts/run_colab.sh`
- `scripts/train_colab.py`

Use the command line and prefer this branch's current versions:

```bash
git checkout <your-branch>
git fetch origin
git merge origin/main

# keep this branch versions for the known conflicted files
git checkout --ours scripts/infer.py scripts/run_colab.sh scripts/train_colab.py

git add scripts/infer.py scripts/run_colab.sh scripts/train_colab.py
git commit -m "Resolve merge conflicts in Colab and inference scripts"
git push
```

Why `--ours` here:
- these scripts include execution-path bootstrapping needed for direct script execution on Colab clones,
- `run_colab.sh` normalizes working directory before installation and training launch.
