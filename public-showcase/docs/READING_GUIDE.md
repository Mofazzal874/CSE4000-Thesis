# Reading guide

1. Read the [method and results](../README.md).
2. Compare the [three-seed recall changes](../results/tab_4_11b_three_seed_replication.csv) with the [matched-pair metrics](../results/tab_4_11_matched_pair_bootstrap.csv). The smallest-target gain does not imply an overall AP gain.
3. Compare [within-domain adaptation and the unseen event](../results/tab_4_13b_adaptation_stages.csv). The unseen-event bootstrap interval includes zero.
4. Inspect the [detection examples](GALLERY.md), including regressions and missed people.
5. Read the [evaluation notes](EVIDENCE.md) before comparing tables that use different splits or thresholds.

Run `python analysis/verify_results.py` from the companion directory to check the exported numbers and file hashes.
