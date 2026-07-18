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

## Public synthetic smoke subset

The repository includes a normalization specification for 27 public OpenHand-Synth line images.
It exists to exercise the data, manifest, inference, and evaluation plumbing without using student
work. It is **not** a representative handwriting benchmark and cannot support a claim about model
quality on children or real classroom documents.

```bash
python -m scripts.prepare_hf_smoke_subset \
  --output-dir data/processed/openhand-synth-smoke
```

The adapter accepts only the configured Hugging Face dataset revision and card licence, then checks
every selected row's index, synthetic style ID, reference, language, source category, dimensions,
and image SHA-256 before publishing `labels.csv`. Its current selection is restricted to English
Faker-generated names and dates; those strings are synthetic, not real identities. Expiring image
URLs are never written to the output. `source.meta.json` and `ATTRIBUTION.md` record the source,
licence, transformation, spec hash, and output hashes.

The normalized images are intentionally ignored by Git. Recreate them from
`configs/datasets/openhand_synth_smoke.json`, and retain the generated attribution file with any
redistributed derivative. The synthetic `writer_id` values describe rendering styles; they do not
identify people.

To exercise the writer-isolated manifest contract after preparation:

```bash
python -m scripts.build_dataset_manifest \
  --dataset-dir data/processed/openhand-synth-smoke \
  --output-dir data/processed/openhand-synth-smoke \
  --dataset-name openhand-synth-smoke \
  --dataset-version 8b5027ab2dc6cc944e0ce7fe37997ce046121e66-selection-v1 \
  --license-id cc-by-4.0 \
  --license-url https://creativecommons.org/licenses/by/4.0/ \
  --sample-type line \
  --train-ratio 0.6 \
  --val-ratio 0.2 \
  --test-ratio 0.2
```

## Public real-handwriting smoke subset

The CSAFE adapter is the first real-handwriting checkpoint. It reads only nine explicitly selected
adult writers' Session 1 PHR pages from the official Figshare archive and publishes 19 deterministic
line crops. The source article and its CC BY 4.0 licence, README, archive, ZIP entries, page images,
crop boxes, references, and output bytes are all pinned. Metadata or content drift causes a hard
failure before `labels.csv` or `source.meta.json` is published.

```bash
python -m scripts.prepare_csafe_smoke_subset \
  --output-dir data/processed/csafe-real-handwriting-smoke
```

The adapter uses HTTP range requests, so it does not download the full multi-gigabyte Session 1
archive. It exports only grayscale line crops, not source pages. The normalized output records the
official citation and licence in `ATTRIBUTION.md`, and identifies source writers only by the
dataset's participant IDs. Keep the attribution with any redistributed derivative.

This set is real adult handwriting, but its line references were split from the known PHR prompt
after one visual pass. It is therefore a pipeline smoke test—not a gold benchmark, not student or
child handwriting, and not evidence of classroom accuracy. An independent human transcription and
crop-boundary review is required before promoting any examples into a gold evaluation set.

To build a writer-isolated manifest after preparation:

```bash
python -m scripts.build_dataset_manifest \
  --dataset-dir data/processed/csafe-real-handwriting-smoke \
  --output-dir data/processed/csafe-real-handwriting-smoke \
  --dataset-name csafe-real-handwriting-smoke \
  --dataset-version figshare-10062203-v2-selection-v1 \
  --license-id CC-BY-4.0 \
  --license-url https://creativecommons.org/licenses/by/4.0/ \
  --sample-type line \
  --train-ratio 0.6 \
  --val-ratio 0.2 \
  --test-ratio 0.2 \
  --min-samples-per-writer 1
```

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

## Protected student-pilot boundary

Private child/student data must first pass the
[protected pilot gate](pilot_governance.md). The gate requires an approved contract, purpose-scoped
consent evidence, repository-external protected storage, opaque sample/writer identifiers, stripped
image metadata, and two blind human transcriptions with independent adjudication on disagreement.

Passing produces `gold_candidate_ready`, not `gold_ready`. Once the candidate population is
owner-approved and the split-access policy is recorded, the separate
[protected evaluation contract](protected_evaluation.md) can freeze a read-only, versioned
validation/test manifest with writer isolation and no training split. Protected inference then
requires a sealed local safetensors artifact and short-lived authorization bound to the exact
model, manifest, device, and inference settings. The public dataset manifest examples above do not
grant permission to process student data.

## Hosted workflow boundary

The `Baseline Predictions` GitHub workflow accepts only a package whose root contains
`manifest.jsonl`, `manifest.meta.json`, and `images/`. Its hosted runner is restricted by an
explicit attestation to public, licensed, de-identified line data. Pseudonymous writer IDs and
references can still be sensitive; do not upload private pilot or student work as a GitHub
artifact. Private evaluation requires protected storage and a controlled/self-hosted execution
path. The repository now provides a fail-closed local contract for that path, but it does not
provide or certify the private storage, access controls, offline inference host, or real pilot data.
