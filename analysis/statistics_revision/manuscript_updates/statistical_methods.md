# Statistical methods

The primary independent sampling unit was the AlphaFold model-parameterization/random-seed trajectory. Recycle snapshots within a trajectory and multiple subunits within one structure were treated as correlated observations. Snapshot-level distributions are reported descriptively as summaries among retained structural snapshots and are not interpreted as thermodynamic populations.

Trajectory-aware uncertainty was estimated by a cluster bootstrap that sampled whole model-seed trajectories with replacement within each condition and retained every qualifying snapshot from each sampled trajectory. Unless otherwise stated, 1,000 bootstrap replicates and random seed 20260803 were used, with percentile 95% confidence intervals. Equal-trajectory estimates first summarized each trajectory and then weighted trajectories equally. Sensitivity analyses retained either the earliest or latest numbered qualifying recycle from each trajectory. The L403A shifted-interface analysis additionally varied the threshold from 11.5 to 15.0 Å in 0.1 Å increments.

Severe G406R mutation-site overlap was defined before analysis as any R406-centered shortest heavy-atom distance <2 Å. Raw and clash-filtered ensembles were analyzed in parallel. Contact frequencies describe protocol sampling frequency, not equilibrium occupancy.
