# Venue and impact audit — 2026-08-17

**Paper:** *Tiny-Aware Assignment and Supervised Sim-to-Real Adaptation for Aerial Human Detection in Disaster Imagery*

**Decision:** Target **IEEE JSTARS** first. The paper is currently a credible, medium-impact applied remote-sensing contribution—not a broad computer-vision breakthrough. WACV 2027 is technically possible but requires an unusually risky 11-day submission sprint and, if accepted, an in-person author registration plus US travel. TGRS, Pattern Recognition, and ISPRS JPRS are stretch targets that require materially broader evidence.

## 1. What the paper actually establishes

### Reproducible positive findings

- Tiny-aware assignment improves recall for **very-tiny (<8 px) humans by +1.62 ± 0.49 percentage points over three seeds**, positive in all seeds. Seed-0 paired bootstrap: +2.08 pp, 95% CI [1.77, 2.38], p<0.001.
- Overall AP50 and F2 are neutral, so this is a **specialist recall reallocation**, not universal detector superiority.
- The trade-off is measurable: recall falls for 8–16 px and 16–32 px objects. This strengthens the scientific honesty but prevents an unqualified “better detector” claim.
- Supervised use of own drone imagery raises drone AP50 from .335 zero-shot to .783 drone-only and .795 with all real data, while C2A AP50 falls only .71 pp.
- Disaster adaptation improves a held-out split from the same source videos substantially, but cross-event evaluation improves only +1.7 pp AP50 and is not significant (p=.47). Therefore, the correct claim is **target-video/domain adaptation**, not general disaster-domain robustness.
- The deployment profile remains practical: 19.57M parameters, 86.7 GFLOPs, about 8.7 ms/image (~115 FPS on the measured machine).
- The work also supplies a leakage-audited scene-disjoint C2A split, three-seed statistics, bootstrap uncertainty, and documented negative results.

### Impact scorecard

| Dimension | Current score | Reason |
|---|---:|---|
| Humanitarian/social relevance | 9/10 | Aerial human detection in disasters is consequential and under-served. |
| Experimental rigor | 8/10 | Scene-disjoint audit, three seeds, bootstrap tests, size strata, external real footage, and honest negatives. |
| Methodological novelty | 5.5/10 | Tiny-aware assignment is a defensible adaptation of known assignment/NWD principles, not a new detection paradigm. FCCG was null and must not be sold as the contribution. |
| Generality | 4.5/10 | The assignment result is centered on C2A/YOLO11m; cross-event disaster transfer is not established. |
| Dataset/protocol value | 6.5/10 if released; 4.5/10 if closed | The own-drone and scene-disjoint protocol can be reusable, but news footage has copyright constraints. |
| Broad CV impact | 4/10 | Insufficient for CVPR/ICCV/ICML main-track claims today. |
| Aerial SAR/remote-sensing impact | 7/10 | Strong domain fit and a useful tiny-object/sim-to-real failure analysis. |

**Overall characterization:** a strong applied field paper with medium scientific impact. It is publishable when framed around measured tiny-human recall allocation plus a rigorous sim-to-real protocol. It is not presently a general SOTA architecture paper.

## 2. Primary recommendation: IEEE JSTARS

### Why it fits

IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing explicitly publishes applied Earth-observation and remote-sensing work and requires new, significant technical content. The UAV disaster-human setting, deployment evaluation, scene-split audit, and sim-to-real study match that audience closely.

A strong recent precedent is **WE-YOLO (JSTARS 2026)**, a wavelet/Mamba tiny-object detector evaluated on AI-TODv2, DIOR-H, and VisDrone. That precedent confirms that modern YOLO-based small-object work is in scope, while also showing why this paper must emphasize its distinctive assignment and disaster sim-to-real evidence rather than generic module addition.

### Financial practicality from Bangladesh

JSTARS is fully open access. Its current APC is **US$1,800**. IEEE’s 2026 geographic-discount list places **Bangladesh in Group B**, normally providing a **50% discount** when all authors meet the institutional-country eligibility rules. That makes the expected APC about **US$900**, subject to IEEE’s final eligibility decision. Adding a non-eligible international coauthor can remove the automatic eligibility, so authorship should be based only on genuine contribution.

### Acceptance assessment

- **Current manuscript, properly written:** realistic but not safe.
- **After the mandatory submission gates below:** good Q1-journal candidate.
- **Main risk:** reviewers may call the method incremental unless the paper cleanly separates the assignment mechanism, its measured size-dependent behavior, and the sim-to-real protocol contribution.

Official sources:

- JSTARS scope and publication information: https://www.grss-ieee.org/publications/journal-of-selected-topics-in-applied-earth-observations-and-remote-sensing/
- IEEE 2026 APC list and geographic discounts: https://open.ieee.org/for-authors/article-processing-charges/
- WE-YOLO DOI: https://doi.org/10.1109/JSTARS.2026.3672925

## 3. Conference option: WACV 2027

### Current deadline reality

As of 2026-08-17, WACV 2027 Round 2 has:

- Paper enrollment: **August 21, 2026 AoE**
- Paper deadline: **August 28, 2026 AoE**
- Supplement: **August 30, 2026 AoE**
- Decision: **October 9, 2026**
- Conference: **January 4–8, 2027**, Disney Springs, Florida

Round 2 has no rebuttal. WACV has Applications, Algorithms, and Evaluations & Datasets tracks. This paper best fits **Applications** or possibly **Evaluations & Datasets**, not Algorithms.

### Topic fit

WACV demonstrably accepts related aerial work. **RealDroneVision (WACV 2026)** bundled a 173,023-image real-drone dataset with an architecture, and **FlyPose (WACV 2026)** addressed aerial person understanding. This confirms topic admissibility, but those papers also set a high evidence bar.

### Bangladesh/travel constraint

WACV requires at least one author to register at the full in-person author rate regardless of presentation format. The 2027 registration price is not yet posted. US visa timing, airfare, accommodation, and registration create materially higher risk than a journal submission from Bangladesh.

### Verdict

WACV is a plausible application-track submission, but the 11-day sprint is high-risk. My recommendation is **do not force WACV unless the paper draft is already nearly complete, the supervisor supports the sprint, and travel/visa funding is credible**. Otherwise, improve the manuscript for JSTARS.

Official sources:

- Dates: https://wacv.thecvf.com/Conferences/2027/Dates
- Author guidance and tracks: https://wacv.thecvf.com/Conferences/2027/AuthorGuidelines
- RealDroneVision: https://openaccess.thecvf.com/content/WACV2026/html/Sabari_RealDroneVision_A_Benchmark_Dataset_and_NOVel_Architecture_for_Enhanced_Small_WACV_2026_paper.html

## 4. Strong future conference fit: IGARSS 2027

IGARSS is an excellent thematic match for UAV Earth observation, disaster monitoring, and risk assessment.

- Submission portal opens: **November 10, 2026**
- Deadline: **January 11, 2027**
- Conference: **July 11–16, 2027**, Reykjavik, Iceland
- A four-page full paper is eligible for IEEE Xplore and the Student Paper Competition.
- The travel-support deadline is also January 11, which matters for a Bangladesh-based student.

The short format limits how much of the assignment, split audit, sim-to-real results, and negative findings can be presented. It is therefore a good conference route for a focused version, but JSTARS is better for the complete study.

Official sources:

- Call: https://2027.ieeeigarss.org/call_for_papers.php
- Dates: https://2027.ieeeigarss.org/important_dates.php

## 5. Stretch journals

### IEEE TGRS

**Fit:** excellent topic fit, much higher novelty/evidence bar. TGRS expects novel and significant remote-sensing methodology with complete experiments. Recent tiny-object papers commonly use several public datasets and stronger architectural/mathematical contributions.

**Current verdict:** stretch/reject-risk. To become credible, add at least one public tiny-object dataset such as AI-TODv2 or TinyPerson, a current transformer detector baseline, and evidence that the assignment idea transfers beyond one YOLO configuration.

TGRS is hybrid: traditional publication avoids an OA APC; optional OA is currently US$2,800. Overlength charges apply beyond ten pages.

Source: https://www.grss-ieee.org/publications/transactions-on-geoscience-remote-sensing/

### Pattern Recognition

**Fit:** possible only if the method is presented as a general pattern-recognition problem. The journal explicitly discourages routine applications of established methods.

**Current verdict:** high stretch. The paper needs formal analysis of why the assignment redistributes recall, experiments across datasets/detectors, and stronger cross-event generalization. Subscription publication has no mandatory publication fee; optional OA is currently US$2,800.

Source: https://www.sciencedirect.com/journal/pattern-recognition

### ISPRS Journal of Photogrammetry and Remote Sensing

**Fit:** thematically excellent but expects broad scientific significance and strong methodological depth.

**Current verdict:** very high stretch. It needs a much larger real cross-event benchmark, public release, multiple datasets and detector families, and a more general method.

Source: https://www.sciencedirect.com/journal/isprs-journal-of-photogrammetry-and-remote-sensing

### Engineering Applications of Artificial Intelligence

**Fit:** good practical AI-engineering fit; public-dataset validation is important.

**Current verdict:** credible second-choice journal, but JSTARS reaches the more appropriate remote-sensing audience. Subscription publication has no mandatory fee; optional OA is currently US$3,040.

Source: https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence

## 6. Lower-risk fallbacks

### IEEE Geoscience and Remote Sensing Letters

Five-page letter format and appropriate scope, but compressing the project would discard much of its strongest evidence. A paper on only the assignment modification risks looking incremental. Use only if a concise letter and lower publication burden are priorities.

### Journal of Real-Time Image Processing

Reasonable if real-time deployment is made central. It is less thematically precise than JSTARS and would require rigorous hardware/latency/export analysis across devices.

### Remote Sensing (MDPI)

The topic fits, but the current APC is CHF 2,700 and the journal does not advertise an automatic Bangladesh nationality discount comparable to IEEE’s geographic program. Keep it as a fallback, not the first target.

## 7. Additional live option: ICCIT 2026

- **CORRECTION (live re-check on 2026-08-17):** the official ICCIT page now gives an **extended
  deadline of August 31, 2026**. The search-engine cache still showed the original July 31 date,
  which caused the earlier audit error.
- ICCIT is a legitimate and unusually close topic fit: Track 2 explicitly includes Computer
  Vision, and Track 10 includes Humanitarian Technologies and Climate Change Applications.
- It is a six-page, double-blind IEEE conference submission. Accepted and presented papers are
  submitted to IEEE Xplore subject to IEEE quality/scope checks.
- A direct precedent exists in ICCIT 2025: *Semi-Supervised YOLO Framework for Real-Time Wildfire
  Smoke Detection from UAV Imagery* (DOI 10.1109/ICCIT68739.2025.11491719).
- ICCIT is materially lower-impact than JSTARS, but lower-cost and lower logistical risk:
  local student registration is BDT 8,000 (IEEE) or BDT 10,000 (non-IEEE), and the conference is
  hybrid in Cox's Bazar.
- A full ICCIT paper and a JSTARS paper cannot be under review concurrently. IEEE permits a later
  journal extension only when it contains substantially more technical information, cites the
  conference paper, and clearly explains the differences.

Full comparison and risk-managed publication route:
`2026-08-17_iccit_vs_jstars_risk_audit.md`.

## 7B. Venues not currently available or not recommended

- **ACCV 2026 / ICPR 2026:** submission deadlines have passed.
- **CVPR/ICCV main conference:** current novelty and breadth are insufficient. The nearest-fit CVPR venue was the 2026 AERO-HPR workshop; no 2027 call is available yet.
- **ICML/NeurIPS:** poor topic/method fit without a fundamental learning contribution.

ICCIT source: https://iccit.org.bd/2026/important-dates/

## 8. Mandatory gates before JSTARS submission

1. Use the exact claim: **very-tiny recall improves +1.62 ± 0.49 pp over three seeds while overall AP50/F2 remain neutral**.
2. Report the 8–16 px and 16–32 px costs prominently; do not hide them.
3. Include the already-run plain **YOLO26m** result as a current contextual baseline, even if broader YOLO26 research is deferred to Lap 5.
4. Distinguish same-video/target-domain adaptation from unseen-event transfer. State that the cross-event result is not significant.
5. Report all three seeds, paired bootstrap intervals, calibration/confidence behavior, size-stratified recall, parameters, FLOPs, and measured latency.
6. Publish code, split-generation logic, scene assignments/hashes, and metric scripts.
7. Release own-drone images/labels only with consent and privacy review. For news footage, release annotation schema, source URLs, timestamps, hashes, and split manifests—not redistributed copyrighted video frames unless licensed.
8. Compare and differentiate from SAFE-Net, RealDroneVision, WE-YOLO, DERNet, RFLA/SimD, and YOLO26/STAL.
9. Frame FCCG as a documented negative experiment, not a contribution.
10. Avoid “robust to real disasters” and “state of the art.” Defensible wording is: **improved target-domain adaptation and very-tiny-human recall under a leakage-audited disaster-aerial protocol**.

## 9. What would raise the ceiling to TGRS / Pattern Recognition

1. Add AI-TODv2 or TinyPerson under the same size-stratified protocol.
2. Test the assignment method on a second detector family or at least a second modern YOLO generation.
3. Add D-FINE/DEIM or RT-DETR as a current transformer baseline.
4. Expand manually labeled unseen-event evaluation across several independent disaster videos and report event-held-out confidence intervals.
5. Demonstrate a principled adaptive policy that gains <8 px recall without simply transferring error to larger size bins.
6. Release a reusable own-drone/disaster benchmark protocol.

## Final ranking

1. **IEEE JSTARS — recommended first submission**
2. **WACV 2027 Applications/Evaluation — viable only as an immediate high-risk sprint**
3. **IGARSS 2027 — strongest conference-theme match, shorter paper**
4. **Engineering Applications of Artificial Intelligence — credible second journal**
5. **IEEE TGRS — stretch after broader experiments**
6. **Pattern Recognition — high stretch after generalization/formalization**
7. **ISPRS JPRS — very high stretch**
8. **GRSL / JRTIP / Remote Sensing — fallback routes**

## Decision record

The best strategy from Bangladesh is to finish a careful JSTARS manuscript while monitoring AERO-HPR 2027 and preparing a focused IGARSS 2027 version only if publication-overlap rules and the journal timeline permit. Do not add international coauthors merely for perceived prestige; authorship must reflect genuine intellectual work, and ineligible affiliations may affect the IEEE geographic APC discount.
