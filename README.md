# MSTT_RUL external-validation releases

## Latest paper-submission release

**HUST final one-shot evaluation and paper submission release**

- Version: `paper-submission-v1.0`
- DOI: `10.5281/zenodo.21758262`
- GitHub release:
  `https://github.com/pengan400-dot/thesis/releases/tag/paper-submission-v1.0`
- HUST confirmation cells: 57
- Confirmation exclusions: 0
- H1 complex-model superiority: not supported
- H2 multiscale superiority: not supported

## Version–DOI map

| Version | Scope | DOI |
|---|---|---|
| v0.1.0-freeze | XJTU pre-unblinding freeze | 10.5281/zenodo.21570017 |
| v0.1.1-amendment | XJTU observation-semantics amendment | 10.5281/zenodo.21571142 |
| v0.2.0-SOH | SOH development freeze | 10.5281/zenodo.21729888 |
| v0.2.1-BIT | BIT structural-feasibility amendment | 10.5281/zenodo.21734477 |
| paper-submission-v1.0 | HUST final one-shot evaluation and paper release | 10.5281/zenodo.21758262 |

The MIT License applies only to author-created code, documentation,
and derived result artifacts. No third-party raw battery datasets are
redistributed.

---

# MSTT RUL Batch-4/5/6 Confirmatory External Validation

This repository contains the preregistered, pre-unblinding software
freeze for confirmatory external validation of recursive lithium-ion
battery remaining-useful-life prediction on Batch-4, Batch-5, and
Batch-6.

## Scientific purpose

The experiment evaluates protocol sensitivity in fixed-cutoff recursive
forecasting, including:

- single-scale versus multi-scale Transformer variants;
- single-step versus multi-step rollout-loss training;
- common target-support enforcement;
- cell-level evaluation;
- nested cross-fit uncertainty calibration;
- independent external validation before external-data unblinding.

## Frozen release

The pre-unblinding release tag is:

```text
v0.1.0-freeze
```

At the time of the freeze:

- three external archive identities were verified;
- twenty-four expected MATLAB filenames were verified;
- archive integrity checks passed;
- external MATLAB payloads had not been parsed;
- external experimental results had not been read;
- nine frozen model-seed jobs were completed;
- twelve nested cross-fit calibration jobs were completed.

The authoritative freeze gate is:

```text
freeze_artifacts/FREEZE_READY.json
```

## External data

The raw Batch-4, Batch-5, and Batch-6 archives are not distributed in
this repository. Users must obtain the datasets from their lawful
original source and comply with the original access conditions.

## Main workflow

```bash
bash manager.sh verify
bash manager.sh inventory
bash manager.sh prepare_dev
bash manager.sh smoke
bash manager.sh train_freeze
bash manager.sh calibrate
bash manager.sh freeze_pack
```

External extraction and evaluation remain blocked until a fixed Git
commit, release tag, Zenodo version DOI, and valid freeze receipt have
been registered.

## Reproducibility records

The project records:

- configuration hashes;
- development-data hashes;
- archive hashes;
- common-support hashes;
- model and scaler hashes;
- environment information;
- per-job training audits;
- calibration predictions and cell-level scores.

## Citation

Citation metadata are provided in `CITATION.cff`.

## License

The source code is released under the MIT License. Dataset licensing
and access conditions remain those of the original dataset providers.

## Repository

https://github.com/pengan400-dot/thesis

## Author

Peng Xue An

ORCID: https://orcid.org/0009-0004-3553-3488

Contact: pengan400@gmail.com
