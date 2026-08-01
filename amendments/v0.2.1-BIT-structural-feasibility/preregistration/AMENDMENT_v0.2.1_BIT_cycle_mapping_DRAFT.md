# AMENDMENT v0.2.1: BIT physical-cycle mapping resolution

**Status:** DRAFT. This text must be committed and registered before any BIT
capacity parsing or model evaluation.

## Governance label

post-BIT-structure-only, pre-model-evaluation amendment

## Evidence available before this amendment

1. The supplied BIT V3 archive contains 73 physical cell directories, including
   72 structurally complete first20/later workbook pairs and one incomplete cell
   (#2).
2. Across all 72 complete cells, the first workbook basename contains
   `first20cycle` or `first20cycles`; the chronological second workbook basename
   contains `cycle` or `cycles`.
3. The first workbook has local `Cycle_Index` 1–21; the later workbook resets to
   local 1–101.
4. The later workbook begins strictly after the first workbook for every complete
   cell, and `Data_Point` resets in the later workbook.
5. The dataset-associated paper analyses early-cycle set sizes from 0 to 20
   cycles. The public dataset/README does not explicitly define how the local
   workbook index 21 should be stitched to the reset later index.

## Amended deterministic mapping

This amendment adopts the explicit filename semantics as the physical-cycle
definition:

- `first20cycle(s)` means the first 20 completed physical cycles.
- First-workbook local cycles 1–20 map to physical cycles 1–20.
- First-workbook local index 21 is retained in the audit trail but excluded as an
  unmapped terminal workbook bucket. It is not used for landmark extraction.
- Later-workbook local cycle `k` maps to physical cycle `20 + k`.
- Consequently, later local 1 maps to physical 21 and later local 101 maps to
  physical 121.

This mapping is disclosed as a governance choice based on filename semantics and
uniform archive structure. It is not claimed to recover undocumented hidden
ground truth.

## Landmark consequence

- Amended primary BIT landmark: physical cycle 70.
- Physical cycles 130 and 190: structurally unavailable and not evaluated.
- Cell #2: excluded because its later workbook is absent.
- Missing official directories #10, #13, #16 and #19 remain recorded as absent.

## Items unchanged

Model architecture and weights, scaler, direct SOH definition, EOL threshold
0.8, H1→H2 order, cohort roles, non-pooling rule, cell-first weighting,
bootstrap/permutation counts, and all statistical testing rules remain unchanged.

## Sources to cite

- Dataset: DOI 10.17632/kw34hhw7xg.3
- Associated paper: DOI 10.1016/j.ensm.2022.05.007
