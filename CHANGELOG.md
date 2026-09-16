# Changelog

## 0.3.0 - 2026-09-16

- Public read-only entry point: rejects mutations at the server, removes write routes from API documentation, and exposes existing reports for viewing/download.
- `evalctl serve --read-only --host 0.0.0.0 --port 6008` provides a portable public launcher. Administration remains on loopback.
- Compact HTML case groups, seed grids, side-by-side checkpoint comparisons, model/task/seed/search filters, pagination, and image density controls.
- Original-image viewer with reference comparison and keyboard navigation.
- Compact PDF v0.3: five seeds per row, typically two single-checkpoint cases per page; multi-checkpoint outputs remain aligned at the same seed. Full prompts, reviews and errors remain in appendices.
- Versioned PDF links and no-store responses at the public entry point avoid stale downloads.
- General TAR/JSONL import wizard, optional targets, configurable fields, and persistent model/case reviews.
- Source archive includes static JSONL examples, fonts with license notices, tests, and a SHA256 manifest. Runtime state, weights, datasets, credentials and deployment history are excluded.

GPU inference requires external weights and runtime dependencies. Krea2 Edit and quality scoring are not represented as validated features.
