# ICCIT 2026 vs IEEE JSTARS — risk audit and publication strategy

**Audit date:** 2026-08-17  
**Purpose:** decide how to protect five to six months of work without overstating acceptance odds.

## Bottom line

1. **ICCIT 2026 is live until August 31, 2026.** The official page was updated from the original
   July 31 deadline; stale search results still show July 31.
2. **The paper is clearly in scope for ICCIT.** The official call explicitly includes Computer
   Vision, Pattern Recognition, Humanitarian Technologies, and Climate Change Applications.
3. **ICCIT offers the safer near-term publication route**, but it is not comparable in academic
   weight to a JSTARS journal article.
4. **JSTARS is the higher-impact and harder target.** The present work is plausible there only
   with careful claims, complete contemporary baselines, and a journal-length experimental story.
5. **No venue can guarantee acceptance.** ICCIT does not publish a current official acceptance
   rate, so assigning a numeric probability would be fabrication.

## Verified ICCIT 2026 facts

| Item | Verified fact |
|---|---|
| Deadline | August 31, 2026 |
| Decision | October 15, 2026 |
| Conference | December 18–20, 2026, Cox's Bazar |
| Mode | Hybrid |
| Paper | Maximum six pages including references, IEEE two-column A4 |
| Review | Double-blind; technical quality, originality, clarity |
| Submission | Microsoft CMT |
| Relevant tracks | Track 2 Computer Vision; Track 10 Humanitarian Technologies / Climate Change |
| Publication | Accepted and presented papers submitted to IEEE Xplore subject to quality/scope |
| Local student fee | BDT 8,000 IEEE member; BDT 10,000 non-member |

Primary sources:

- https://iccit.org.bd/2026/important-dates/
- https://iccit.org.bd/2026/online-paper-submission-system/
- https://iccit.org.bd/2026/scope/
- https://iccit.org.bd/2026/registration/
- https://cmt3.research.microsoft.com/ICCITconf2026

## Evidence that the topic will be considered

ICCIT 2025 published *Semi-Supervised YOLO Framework for Real-Time Wildfire Smoke Detection from
UAV Imagery*. It used 738 labelled and 1,887 unlabelled UAV images, compared YOLO/SSL methods,
reported mAP50 and mAP50–95, and framed the system for disaster response. It is now in IEEE Xplore:

- https://doi.org/10.1109/ICCIT68739.2025.11491719

This is unusually close evidence of topic fit. The current human-detection paper has additional
rigor that the abstract of that precedent does not advertise: scene-leakage audit, three seeds,
paired bootstrap inference, explicit <8 px evaluation, own-drone adaptation, and cross-event
failure analysis.

Therefore, **scope mismatch is not the main ICCIT rejection risk**. The risks are six-page clarity,
method novelty, inconsistent claims, and weak presentation of the neutral overall-AP result.

## What the venue comparison really means

| Dimension | ICCIT 2026 | IEEE JSTARS |
|---|---|---|
| Type | Broad IEEE conference proceedings | Specialist remote-sensing journal |
| Paper length | 6 pages total | Journal-length |
| Topic fit | Explicit CV + humanitarian tracks | Strong applied Earth-observation/UAV fit |
| Review bar | Originality/technical quality; broad audience | New and significant field contribution |
| Current selectivity | No official 2026 acceptance rate published | No reliable public acceptance rate found |
| Academic weight | Respectable regional IEEE conference paper | Q1 field-journal-level publication |
| Indexing | Xplore submission after acceptance/presentation and quality checks | IEEE Xplore journal article |
| Cost from Bangladesh | BDT 8k/10k student | US$1,800 APC; expected 50% geographic discount if eligible |
| Time to decision | Fixed: October 15 | Variable journal review/revision cycle |
| Travel | Hybrid, domestic venue | None |
| Rejection consequence | Can revise toward another venue | Can revise/resubmit elsewhere, but slower |
| Best use | Secure a focused preliminary archival result | Publish the complete scientific story |

ICCIT is broad and high-volume. Its 2024 proceedings spanned several thousand pages and included
many application/comparative deep-learning papers. That supports a lower-bar assessment relative
to JSTARS, but it does **not** supply an acceptance probability because the number of submissions
has not been officially reported.

JSTARS has recent small-UAV-detection articles such as αS-YOLO (2025), which reported about
1.0 mAP and 1.1 mAP50 improvement over YOLOv8 on VisDrone while proposing both a context module
and a new localization loss:

- https://doi.org/10.1109/JSTARS.2025.3539873

This shows that modest effect sizes can be publishable in JSTARS, but only when accompanied by a
clear mechanism, accepted benchmark positioning, complete ablations, and journal-quality evidence.

## Rejection-risk assessment for this exact project

### ICCIT

**Positive factors**

- Exact scope and a direct 2025 UAV-disaster-YOLO precedent.
- More rigorous statistics than many six-page application papers.
- Local humanitarian relevance and real field data.
- Contemporary YOLO26 anchor already available.
- A negative overall-AP finding can be framed as a size-specific operating trade-off.

**Risks**

- +1.62 pp <8 px recall may look small without counts, confidence intervals and operational meaning.
- Overall AP50/F2 neutrality makes a “better detector” title indefensible.
- Too many storylines—assignment, leakage, sim-to-real, FCCG, disaster video—will collapse in six
  pages.
- If the method is described merely as “NWD added to YOLO,” novelty will look incremental.

**Assessment:** credible submission with a materially lower bar than JSTARS, but not guaranteed.
The strongest ICCIT paper is narrow and disciplined.

### JSTARS

**Positive factors**

- Strong domain alignment.
- Scene-disjoint protocol, real UAV data and tiny-size statistics are meaningful.
- Honest cross-event and negative results improve credibility.
- More complete story than a generic multi-module YOLO paper.

**Risks**

- The main assignment change is incremental relative to RFLA/SimD/STAL/NWD literature.
- Only one core public disaster dataset demonstrates the method.
- Larger-object recall costs weaken the method.
- Unseen-disaster generalization was not statistically established.
- The paper must compete with 2025–26 multi-dataset methods and attention/frequency detectors.

**Assessment:** plausible but appreciably higher rejection risk. It becomes substantially stronger
with one additional public tiny-object dataset or a second detector-family transfer experiment.

## Recommended risk-managed route

Because the user's priority is to avoid ending with no publication, the recommended route is:

### Stage 1 — submit a focused ICCIT paper by August 31

Provisional title:

> **Tiny-Aware Label Assignment for Very-Tiny Human Detection in UAV Disaster Imagery**

Keep only:

1. The scene-disjoint C2A protocol and measured tiny-object distribution.
2. The tiny-aware assignment mechanism.
3. Three-seed <8 px recall result and bootstrap interval.
4. Overall AP/F2 neutrality and larger-size trade-off.
5. YOLO11m control, D2, YOLO26m and the most essential published comparators.
6. One compact own-drone zero-shot check if space permits.

Do **not** make FCCG a contribution. Do not claim disaster robustness or SOTA.

Suggested six-page budget:

- 0.7 page introduction/contributions
- 0.5 related work
- 1.0 method
- 0.7 protocol/data audit
- 2.3 results, ablation and significance
- 0.3 limitations/conclusion
- references within remaining space

### Stage 2 — develop a substantially expanded JSTARS article

The journal extension should add substantial technical information:

1. Full own-drone, campus and disaster sim-to-real ladder.
2. Cross-event evaluation and the documented non-generalization result.
3. Calibration, false-positive and altitude-stratified analyses.
4. Full efficiency/deployment study.
5. A new transfer experiment on AI-TODv2/TinyPerson or a second detector family.
6. Complete negative-result and architecture-selection analysis.
7. Public split/code/protocol release.

The journal must cite the ICCIT paper and explicitly enumerate the new material.

IEEE policy permits a conference paper to evolve into a journal article only when the journal
contains substantially more technical information, cites the conference version, and clearly
states the differences:

- https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/ethical-requirements/

**Never submit ICCIT and JSTARS concurrently.**

## Alternative: direct JSTARS

Choose direct JSTARS instead if:

- the supervisor values journal rank much more than a near-term publication;
- the APC is funded;
- another public-dataset or detector-transfer experiment can be completed;
- the team accepts a slower review cycle and higher rejection risk.

Direct JSTARS preserves all novelty in one manuscript. It has the higher ceiling, but is not the
best route if the overriding goal is a near-term, lower-risk archival publication.

## Final decision

For protecting the work while retaining a journal route:

> **Submit a focused preliminary paper to ICCIT 2026, then build a genuinely expanded JSTARS
> article rather than sending the same paper twice.**

This recommendation is based on verified topic fit, deadline, cost and publication policy—not on
an invented acceptance guarantee.
