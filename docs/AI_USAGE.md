# AI usage log

Binding rules: see CLAUDE.md at the repo root.

| Date | Tool | What | Files |
|------|------|------|-------|
| 2026-11-04 | Claude Code | Scaffold: directory layout, header-only core/ext/tests files, configs (author-specified values only), pyproject, README, .gitignore, docs. No core logic. | all initial files |
| 2026-11-04 | Claude Code | Env check and dataset download scripts (non-core) | scripts/check_env.py, scripts/download_mipnerf360.sh |
| 2026-11-05 | Claude Code | macOS/Windows env support: check_env rewrite, PowerShell download script, .gitattributes, README setup | scripts/check_env.py, scripts/download_mipnerf360.ps1, .gitattributes, README.md |
| 2026-11-17 | Claude Code | Policy update: core is now defined as all of src/splat/ except assisted/ (core/ folder was removed); added a rule against giving corrections to human-written code | CLAUDE.md, README.md, docs/AI_USAGE.md |
