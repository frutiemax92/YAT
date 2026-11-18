# Rules for agents working in this repository

## Never commit or push training configs (`config*.yaml`) or credentials

- **Never** `git add`, commit, or push any `config*.yaml` file (new or existing), unless the user explicitly asks for that exact file to be committed and it has been verified to contain no secrets or personal/local paths.
- Training configs (`config_*.yaml`) may contain Cloudflare R2 credentials (`r2_endpoint`, `r2_access_key`, `r2_secret_key`, `r2_bucket_name`), local dataset paths, or personal training settings that must stay local and never reach git history or the remote.
- Never stage `civitai_key`, `r2_credentials.yaml`, or any other key, token, or password file.
- Before every commit, run `git diff --cached` and verify:
  1. No `config*.yaml` file is staged.
  2. No `r2_access_key`, `r2_secret_key`, `r2_endpoint`, `civitai_key`, or other secret appears in the staged diff.
  If a check fails, unstage the file immediately and inform the user.
- Do not copy secrets into tracked files (scripts, docs, READMEs, tests), commit messages, or PR descriptions.

## Commit and push every feature and fix

Each time a feature is added or a bug is fixed, commit it and push it to the remote right away, without waiting to be asked:

- One commit per feature or fix, with a short imperative commit message matching the existing history (e.g. `Add text embedding precompute`, `Fix local shard getter`).
- Stage only the files that belong to that change, by explicit filename. **Never** use `git add -A`, `git add .`, or `git commit -a` (the working tree contains untracked datasets, archives, and local `config*.yaml` files that must not be committed).
- Run `git diff --cached` and apply the credential and config checks above before committing.
- Then `git push`. If the push is rejected, report it instead of force-pushing.
- Only commit finished, tested changes.
