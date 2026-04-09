# Phase 5 - supervisor summary

## Status

Current paper-driven objective reproduction status: `partial`.

## What was achieved
- The original POP909 training path now runs locally and on the cluster with GPU.
- The objective metric protocol from section 5.2 of the paper was implemented in the project.
- A first checkpoint-level comparison already shows the same qualitative disentanglement pattern reported by the paper:
  - pitch perturbations affect `zchd` more than `ztxt`
  - rhythm perturbations affect `ztxt` more than `zchd`
  - octave transposition leaves `zchd` unchanged in the first smoke evaluation

## Why the status is still partial
The current comparison was executed on a short proof checkpoint, not yet on a more representative run closer to the paper's full training setting. So the objective direction is encouraging, but the final paper-level claim should remain conservative for now.

## Next useful step
Run one more representative training experiment and evaluate it with the same objective metric script. That would support a much stronger comparison with the paper.
