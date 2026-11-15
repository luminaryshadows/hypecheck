# Security and deployment boundaries

HypeCheck is a static, read-only historical research dashboard, not a trading or account service.

## Controls

- The public app executes no Python, model downloads, commands, orders, uploads or paid APIs. It has no login, database, analytics, cookies or secrets.
- Post text and all report values are rendered with DOM `textContent`, not HTML. Only HTTPS links to exact `x.com`/`twitter.com` hostnames are clickable; links use `noopener noreferrer`.
- The frontend uses same-origin scripts/styles/data with no CDN dependencies and a restrictive meta Content Security Policy. No inline scripts or inline event handlers are required.
- The local preview binds only to `127.0.0.1`, serves an explicit list of public demo and archive assets, rejects mutations, and does not expose source directories, `.git`, caches or directory listings. It adds CSP, frame denial, MIME sniff prevention, no-referrer and permissions restrictions.
- The training pipeline pins Hugging Face dataset and model revisions, verifies the dataset's upstream LFS SHA-256, requires `safetensors`, sets `trust_remote_code=False` and requests public model files with `token=False`. It never loads pickle/joblib checkpoints or `training_args.bin`.
- Price downloads use hard-coded HTTPS public endpoints, fixed coin allowlists, timeouts and bounded retry/pagination. Missing days, invalid OHLC, duplicate dates or non-finite values stop the run rather than being silently filled.
- Raw post archives, downloaded model weights, environment files and local caches are excluded from version control. The current replay uses up to two short excerpts per coin and date with links to X. The five-coin archive contains earlier excerpts.

## What these controls do not establish

GitHub Pages controls HTTP response headers. The meta CSP cannot provide `frame-ancestors`, HSTS or every header added by the loopback preview. Do not describe the public deployment as having every local header. HTTPS is enabled at the hosting layer.

Checksums detect changes against the pinned source or recorded local artifacts; they do not prove the source data is truthful. SHA provenance is not an independent audit of X posts or exchange bars.

No security review proves the absence of every vulnerability. Dependency audit results and known limitations must be stated separately from passing application tests. The public report intentionally contains public historical text and predictions; it is not confidential.

## Verification

```bash
python -m unittest -v test_web.py
node docs/app.js --self-test
```

After generating data, also run `python -m unittest -v test_pipeline.py`.

Never add arbitrary URL fetches, user-uploaded checkpoints, credentials or order execution to this demo without a new threat model and review.

## Submission hardening

- Cached sentiment JSON is checked for exact labels, numeric types, finite values, [0,1] bounds and probability sum before use. The cache is still local mutable data: these checks do not authenticate its origin. The current notebook checks its cached scores against a pinned hash by default and offers fresh inference with the pinned encoder.
- The loopback server refuses symlink targets and paths resolving outside its fixed docs root, even on allowlisted routes. A regression test attempts an allowlisted symlink to a private file.
- Notebook subprocess failures stop execution; the local runner records errors rather than marking failed commands as successful cells.
- Notebook regeneration preserves outputs only when the entire ordered code-cell sequence matches. Changed code invalidates saved execution outputs.
- The dependency-audit JSON predates this hardening pass. It is historical evidence, not a new vulnerability scan or a clean bill of health.
