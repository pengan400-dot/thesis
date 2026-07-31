# Engineering validation report

Date: 2026-07-27

This report documents code-path validation only. Synthetic outputs are not
scientific results and are excluded from every freeze and result package.
No BIT model code was run and no BIT capacity trajectory was accessed.

## Passed checks

- Python compilation for all source, script, and test files.
- Bash syntax validation for `manager.sh`.
- Fifteen unit tests:
  - SOH formula direction and first-five reference rules;
  - rejection of pre-normalized input mislabeled as Ah;
  - exact and Monte Carlo sign-flip contracts;
  - fixed-sequence H1/H2 gate behavior;
  - BIT structure-only output redaction;
  - full and single-scale PyTorch forward passes.
- Independent analytical and PyTorch parameter counts:
  - full K5: 74,405;
  - single-scale K5: 69,789.
- Synthetic 16-cell XJTU development smoke:
  - eight B1 and eight B3 cells passed normalization;
  - two-epoch full-K5 training and checkpoint/scaler save passed;
  - one outer conformal fold, one seed, and cutoffs 70/130/190 passed.
- Real HNEI archive structure and parser:
  - archive SHA-256:
    `a6c33eaee5edbf6288dea4bf8ce7513b0fb1c27579c8ae78882dd1279f54a901`;
  - 14/14 primitive-only pickle members passed the opcode gate;
  - 14/14 cells passed the new first-five median SOH parser;
  - first-five median references ranged from 2.585 to 2.650 Ah;
  - all 14 EOL crossings remained interval-determined.

## Deliberate non-tests

- Formal 300-epoch XJTU training was not run because the real 16 XJTU input
  curves are not present in this workspace.
- Formal conformal quantiles were not generated.
- HNEI model performance was not generated.
- BIT schema remains unresolved until the user runs structure-only preflight.
- No BIT model evaluator is included in this release.
