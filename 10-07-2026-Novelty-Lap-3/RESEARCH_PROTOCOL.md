# Research Protocol — verify before propose (standing rule since 2026-07-10)

User rule (2026-07-10, this lap's founding instruction): **every time Claude proposes a method,
paper, or module, it must FIRST verify it on the live web via search agents and curate primary
sources.** No memory-only citations for anything published after ~2024. This exists because a
Google-search AI fed the project a mix of real 2025/26 work and fabricated names, and because
Claude's own cutoff (Jan 2026) misses the newest results.

## Procedure (what Claude does, every time)
1. **New idea or external claim → spawn web-search agent(s)** (WebSearch + WebFetch; loaded via
   ToolSearch when deferred) BEFORE proposing anything. One agent per topic lane; parallel.
2. **Verification bar:** a claim counts as verified only with a primary source — arXiv abs page,
   DOI/publisher page, official docs, or official GitHub repo (repo URLs must be fetch-verified).
   Medium/blog/SEO pages are secondary color, never evidence.
3. **Triage verdicts:** `VERIFIED` / `PARTLY-REAL (GARBLED)` / `UNFINDABLE (suspected fabrication)`.
   An honest UNFINDABLE is a valid, valuable result. Never invent a citation or arXiv ID.
4. **Every proposal doc must state:** primary-source table · which measured need (N1–N8) it
   answers · honest expected effect size (with the ~50% stacking discount) · cost on our PC fleet ·
   prior-art distance (why it is not plug-and-play).
5. **Outputs are dated md files** in the current lap folder's `docs/`; raw agent reports go to a
   `*_web_verification_log.md` for traceability. Committing: Claude presents a review summary
   (changed files + suggested message) and the USER commits/approves — no self-initiated commits
   (user rule 2026-07-12).
6. **Falsified-locally list** (do not re-propose without NEW evidence):
   - MuSGD → diverged on our P2 config (2026-06; AdamW lr0=0.001 pinned)
   - NWD as IoU-replacement loss → gate G2 CLOSED (+0.73 VT-recall @α=.5, does not scale)
   - Mamba / AtrousSSM → 2 genuine negative runs, 2.8× latency
   - ECA → null · copy-paste aug → negative · plain CBAM/P2 → done, is the baseline, not novelty
7. **Known-fabricated list** (from the 2026-07-10 audit; see catalog doc for the full table):
   items claimed by outside AI chats that no primary source supports must be named as such in any
   doc that mentions them, so they never re-enter planning as "real".

## Standing quality rules (inherited, still binding)
- Feasibility check on OUR stack (Ultralytics YAML pipeline, Windows, 16 GB protocol GPU) before
  proposing; register-inject-verify pattern with pickle-roundtrip + module-active assertions.
- Every figure/number traceable to a file; effect sizes stated honestly; PowerShell syntax for all
  run commands; check files exist on the target PC before issuing commands.
