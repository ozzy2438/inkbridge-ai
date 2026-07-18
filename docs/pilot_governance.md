# Protected Student-Handwriting Pilot Gate

## What this gate proves

InkBridge refuses to ingest a private child/student gold candidate unless its package supplies
machine-checkable evidence for approval, consent scope, de-identification, protected storage,
retention/deletion, and two independent human transcriptions. The validator emits only hashes,
counts, and aggregate review metrics.

Passing this gate is **not legal advice, a privacy certification, or permission to run a pilot**.
The named data controller, educator, and privacy owner remain responsible for approvals and the
facts asserted by the contract. The software can verify a signed-off field and a dataset property;
it cannot verify that a consent conversation was valid.

## Why these controls exist

The Victorian Department of Education warns that generative-AI tools can request or receive
student personal, sensitive, or health information and that uploads can create privacy and data
protection risks. Private student data therefore cannot use the public demo or GitHub-hosted model
workflows.

The Australian Privacy Principles require reasonable safeguards against misuse, loss, unauthorised
access, modification, and disclosure, and require destruction or de-identification when personal
information is no longer needed. OAIC guidance also says a young person's capacity to consent is
case-specific; a parent or guardian may need to consent, and the child should still be involved as
far as practicable.

Official references:

- [Victorian Department of Education: Protecting privacy and personal data](https://www2.education.vic.gov.au/pal/generative-artificial-intelligence/guidance/protecting-privacy-and-personal-data)
- [OAIC: Australian Privacy Principles](https://www.oaic.gov.au/privacy/australian-privacy-principles/read-the-australian-privacy-principles)
- [OAIC APP 11: Security, destruction, and de-identification](https://www.oaic.gov.au/privacy/australian-privacy-principles/australian-privacy-principles-guidelines/chapter-11-app-11-security-of-personal-information)
- [OAIC: Consent capacity for children and young people](https://www.oaic.gov.au/privacy/australian-privacy-principles/australian-privacy-principles-guidelines/chapter-b-key-concepts)

## Package layout

The real package stays outside the Git checkout in approved private storage:

```text
/protected/inkbridge/pilot-001/
├── images/
│   ├── sample-<opaque-hex>.png
│   └── ...
├── labels.csv
├── annotation_review.csv
├── pilot_intake.json
├── protected_storage_control.json
└── protected_lifecycle_events.jsonl
```

First prepare and owner-approve the repository-external storage boundary described in
[protected storage and lifecycle evidence](protected_storage_lifecycle.md). The intake gate now
requires owner-only POSIX permissions, a current read-only storage control, a valid hash-chained
lifecycle ledger, no open consent withdrawal, and no deletion deadline breach.

Copy `configs/datasets/protected_student_pilot.template.json` into protected storage and replace
every pending/placeholder field with an approved internal reference or verified setting. The
template deliberately fails validation and contains no approval.

Run the gate locally in the controlled environment:

```bash
python -m scripts.validate_protected_pilot \
  --contract /protected/inkbridge/pilot-001/pilot_intake.json \
  --dataset-dir /protected/inkbridge/pilot-001
```

The result, `pilot_intake.audit.json`, is stored beside the protected package. It contains no
transcripts, writer IDs, annotator IDs, or source images. Do not upload the package or audit to
GitHub Actions; the audit hashes still describe sensitive data and remain protected provenance.

## Fail-closed checks

- Dataset, contract, review file, and audit must be outside the Git repository.
- The storage control and lifecycle ledger must pass the protected storage/IAM boundary; all tree
  entries deny group/other access and no symlink is allowed.
- Storage approval is valid for at most 90 days and retention settings exactly match the pilot
  contract.
- An open consent withdrawal or deletion deadline breach blocks intake and all downstream gates.
- Contract and review files cannot be symlinks.
- Privacy and educator approval must be `approved`; consent must be active and purpose-specific.
- Consent authority and capacity assessment are explicit; consent records stay outside the data.
- Publication, public-repository upload, GitHub Actions, third-party AI upload, and automated
  educational decisions are disabled.
- Raw retention is at most 30 days, normalized-candidate retention at most 180 days, and verified
  deletion/de-identification after withdrawal is at most 30 days. These are conservative InkBridge
  pilot-policy ceilings, not statutory periods; an approved pilot may set shorter periods.
- Only opaque filenames/writer IDs and coarse age bands are allowed.
- Indexed images exactly match `images/`, contain no EXIF/text metadata, and cannot be symlinks.
- Every final transcript has two blind passes and a crop review.
- Disagreement requires an independent adjudicator, controlled reason, and recorded time.
- At least three writers are required so later validation/test groups can be writer-isolated.
- The evaluation-set version, deterministic seed, validation/test writer ratios, and minimum
  samples per writer are approved in the contract before the split is revealed.

## What remains after a pass

The audit status is `gold_candidate_ready`, while `gold_ready` remains false. The implemented
[protected evaluation gate](protected_evaluation.md) then:

1. revalidates the approved package against its persisted intake audit;
2. creates a versioned manifest with disjoint validation and locked-test writers;
3. writes no training split and makes the manifest and metadata read-only;
4. binds a short-lived approval to a sealed local model and exact inference settings;
5. rejects hosted CI, reference-bearing prediction exports, and external-AI execution;
6. joins protected references only in memory and persists aggregate metrics only.

The code now verifies an owner-only self-hosted POSIX boundary and hash-chained withdrawal/primary-
and backup-deletion evidence. Production still needs independent proof for encryption, provider IAM,
network controls, immutable audit-log retention, and actual backup erasure. The remaining evidence
step is an authorised real pilot with independently reviewed data. A frozen pilot set still reports
`gold_ready: false`; size, coverage, owner acceptance, and evaluation evidence must be reviewed
before any final gold or release claim.

No real student image, transcript, manifest, consent record, or annotation review is committed to
this public repository.
