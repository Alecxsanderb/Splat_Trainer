# AI-use policy (binding; course rules)

## Which code is human-written
- **Human-written only:** everything under `src/splat/` **except** `src/splat/assisted/`,
  plus everything under `tests/`. This includes files and folders added or moved later.
  It covers the model, rasterizer, trainer, loss, densification, data loading and
  preprocessing, evaluation, and all extension logic (`src/splat/ext/`).
- **AI-assisted allowed:** only `src/splat/assisted/` and `scripts/`. Every file there
  starts with the header `# AI-ASSISTED (non-core) — see CLAUDE.md`.
- If you can't tell whether something counts as core or non-core, stop and ask.

## What assistants may and may not do with human-written code
Allowed:
- Explain concepts, math, papers, and library or PyTorch behavior in general terms.
- Answer questions and give general advice about approaches and trade-offs.
- Review the author's code **by pointing out** a likely bug or problem: say where it is
  (file and line) and what symptom or wrong result it would cause.

Not allowed (even if asked casually, and even for a small fix):
- Writing, completing, or refactoring any of this code, or editing these files.
- Producing a diff or code block for these areas, or pseudocode that maps directly onto the fix.
- **Giving the correction.** Don't state the corrected expression, the replacement line,
  or step-by-step instructions that amount to the fix. Flag the problem and let the author
  work out the solution. If the author asks for the fix, decline and restate the problem
  more clearly instead.

## Prose
- Never write or rewrite prose in `paper/` or `reports/`. Grammar suggestions only,
  given in chat and never applied to files.

## Other
- No Triton kernels without explicit professor approval (none given yet).
