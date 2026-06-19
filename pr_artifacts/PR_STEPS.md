# PR creation steps

The current machine has no GitHub write credential for `dasimawudi/psd` and no accessible fork, so I could not create the PR directly.

Current prepared branch:

```bash
best/disk-center-region 7748c32 Organize best region model branches
```

If you create/access a fork, for example `YOUR_USER/psd`, run:

```bash
git remote add fork https://github.com/YOUR_USER/psd.git
GIT_ASKPASS= SSH_ASKPASS= git push -u fork best/disk-center-region:best/disk-center-region
```

Then open this URL, replacing `YOUR_USER`:

```text
https://github.com/dasimawudi/psd/compare/main...YOUR_USER:psd:best/disk-center-region?expand=1
```

PR title:

```text
Organize best region model branches and disk-center baseline
```

Use `pr_artifacts/PR_BODY.md` as the PR body.

If you cannot push a fork from this machine, apply the prepared patch in another clone:

```bash
git clone https://github.com/YOUR_USER/psd.git
cd psd
git checkout -b best/disk-center-region origin/main
git am /data-ssd/libo/psd/pr_artifacts/best-region-model-branches.patch
git push -u origin best/disk-center-region
```
