# Formal v0.2.1 BIT structural-feasibility amendment

Governance label: post-BIT-structure-only, pre-model-evaluation amendment.

The supplied BIT V3 archive cannot represent physical cycles 130 or 190.
Physical cycle 70 is the sole frozen BIT landmark. The filename-semantic
offset-20 mapping is:

- first20 local 1..20 -> physical 1..20
- first20 local 21 -> excluded terminal workbook bucket
- later local k -> physical 20+k

The archive contains 73 observed cell directories. Seventy-two paired cells are
included: 55 arbitrary-use and 17 fixed-profile. BIT_#2 is excluded because its
later workbook is absent. Official IDs 10, 13, 16, and 19 are absent from the
supplied archive.

Model weights, scaler, SOH definition, EOL threshold, H1->H2 order, cohort
roles, cell-first weighting, non-pooling rule, and inferential settings remain
unchanged. This amendment must be registered before any BIT capacity parsing or
model output.
