# EXP36-v3: Prefix-Replay Feasibility Audit

## Role

Source-only prefix-replay feasibility experiment.

This experiment is not a true leave-one-domain-out evaluation.

NASA follows the historical EXP27 recursive protocol.
CALCE-CS2 and XJTU-B2 follow the historical EXP28 recursive protocol.

## Frozen status

`STOP_BEFORE_TRAIL_DGR_TRAINING`

The preregistered feasibility gate was not passed.
No subsequent TRAIL-DG-R training was performed after this stopping decision.

## Historical reproduction

Maximum absolute deviation from the frozen historical recursive results:

`9.887923813068e-17 Ah`

## Main results

Equal-domain physical-cell mean SOH RMSE:

| Method | SOH RMSE |
|---|---:|
| Local | 0.1868758103 |
| Always-Lite | 0.0817908889 |
| Always-Full | 0.0906097329 |
| Prefix-Replay v3 | 0.1597497199 |

Prefix-Replay v3 relative change versus Always-Lite:

`+95.31%`

Additional results:

- Domain wins versus Always-Lite: `0/3`
- Physical cells where Prefix-Replay was better than Lite: `3/23`
- Prefix-Replay FCAR: `0.065217`
- Always-Lite FCAR: `0.173913`
- Local fallback rate: `0.444444`

Physical-cell stratified bootstrap:

`Delta(Replay - Lite) = +0.0779588310`

`95% CI = [+0.0124404432, +0.1433872709]`

## Interpretation

Prefix replay reduced harmful neural adoption, but did not reliably
predict sustained long-horizon neural advantage.

Its increased conservatism substantially reduced useful neural-model
adoption, and therefore the predefined feasibility criteria were not met.

## Frozen evidence archive

Archive:

`EXP36_PREFIX_REPLAY_V3_FINAL_20260810_230910.tar.gz`

SHA256:

`213fc1584ca16bf3d0870bd622d31b922eaa331c37c2fced5452f98c1d7f024e`

The complete frozen archive is distributed separately as a GitHub
Release asset rather than being committed to normal Git history.
