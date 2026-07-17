# Dataset Contract

Training and evaluation data must be normalized before it enters an InkBridge experiment. This
repository does not download restricted datasets, accept licence terms, or treat private student
work as training data automatically.

## Normalized input

A dataset directory contains image files and a CSV label index:

```text
dataset/
├── images/
│   ├── page-001.png
│   └── page-002.png
└── labels.csv
```

`labels.csv` must have `filename,text,writer_id` columns. An optional `slices` column contains
pipe-delimited evaluation tags such as `age_8|faint_pencil`; tags are de-duplicated and sorted.
Filenames must be basenames under `images/`; absolute paths, parent traversal, missing files,
duplicate filenames, empty text, and empty writer IDs are rejected.

## Build the manifest

```bash
python -m scripts.build_dataset_manifest \
  --dataset-dir /protected/data/gnhk-normalized \
  --output-dir /protected/data/gnhk-normalized \
  --dataset-name gnhk \
  --dataset-version v1 \
  --license-id CC-BY-4.0 \
  --license-url https://creativecommons.org/licenses/by/4.0/ \
  --sample-type line
```

The command writes `manifest.jsonl` and `manifest.meta.json`. Every record contains a stable sample
ID, source-file SHA-256, relative image path, reference, pseudonymous writer hash, dataset version,
licence provenance, sample type, slices, and assigned split. Metadata includes the source-label and
manifest hashes, split parameters/counts, duplicate-source count, and writer-isolation assertion.
Writing the files beside `images/` creates the self-contained package expected by baseline
inference. Use a separate output directory when only the manifest is needed.

Writers below `--min-samples-per-writer` cause a hard failure instead of being silently discarded.
Requested train, validation, and test splits must each receive at least one writer. The same seed and
input reproduce the same assignment.

The writer hash is pseudonymisation, not anonymisation. Manifests containing private references or
linkable image paths must remain in protected storage with the same retention and access controls as
the source dataset; do not commit pilot/student manifests to this repository.

Public dataset licence fields in configuration are provenance defaults, not legal approval for a
particular use. Verify the source terms before downloading, training, redistributing, or using a
dataset commercially.

## Hosted workflow boundary

The `Baseline Predictions` GitHub workflow accepts only a package whose root contains
`manifest.jsonl`, `manifest.meta.json`, and `images/`. Its hosted runner is restricted by an
explicit attestation to public, licensed, de-identified line data. Pseudonymous writer IDs and
references can still be sensitive; do not upload private pilot or student work as a GitHub
artifact. Private evaluation requires protected storage and a controlled/self-hosted execution
path, which this repository does not yet provide.
