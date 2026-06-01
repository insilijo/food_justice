# food_justice Roadmap

## Current State (2026-03-23)

### Working well
- Static-site build pattern is documented.
- The repo has clear scripts/docs structure and recent activity around metro-specific updates.

### Quality assessment
- Overall quality: active data/publication repo with unclear engineering boundaries.
- The README is narrow and build-focused; it does not explain product scope, validation standards, or maintenance workflow.
- The repo is large and likely output-heavy, which raises review and reproducibility costs.

### Highest risks
- Build assumptions live more in operator memory than in durable documentation.
- Large data/output volume can mask regressions.
- It is unclear which generated artifacts are canonical and which are disposable intermediates.

## Priority Roadmap

### Phase 1 — Scope and Reproducibility
- [ ] Expand project docs beyond the one build recipe.
- [ ] Define canonical input datasets, metro config sources, and output contracts.
- [ ] Add a clean "rebuild from scratch" workflow.

### Phase 2 — Validation
- [ ] Add metro-level regression checks and manifest generation.
- [ ] Validate core summary outputs against known reference values for at least one city.
- [ ] Add publication-facing QA for map completeness and broken asset links.

### Phase 3 — Maintenance
- [ ] Separate generated artifacts from authored logic more clearly.
- [ ] Add an update cadence and ownership checklist for new city releases.

## Recommendation

The repo looks workable but under-documented. It needs operational discipline more than new analytical features.
