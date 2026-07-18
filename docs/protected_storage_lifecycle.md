# Protected Storage, IAM, and Lifecycle Evidence

This contract prepares a repository-external self-hosted POSIX boundary for a private handwriting
pilot and makes consent withdrawal/deletion state part of every later evaluation binding. It is a
technical control, not proof of legal authority or a substitute for cloud/storage audit evidence.

## What is verified and what is attested

The validator directly verifies that:

- the protected root is outside Git and is not a symbolic link;
- every descendant is owned by the current approved POSIX user/group;
- group and other permission bits are zero across the tree;
- the storage control is a non-symlink, read-only root file;
- the operator UID/GID, pilot ID, storage ID, retention limits, and approval window match;
- the accountable owner, runtime operator, and deletion-verification owner roles match their
  contract-bound responsibilities;
- the lifecycle ledger is a private regular file with a valid event sequence and SHA-256 chain;
- no consent withdrawal is open and no deletion deadline has been breached.

The control file separately records owner attestations for encryption at rest/in transit, named
least privilege, infrastructure audit logging, backup retention scope, network isolation, public
access, and cross-border disclosure. POSIX mode checks cannot independently prove those controls,
FileVault/KMS state, host ACLs outside mode bits, firewall policy, backup erasure, or the authority
of the named roles. Retain provider/host evidence beside the consent system, not in Git.

## 1. Prepare the protected tree

Stage the package outside the repository, then run:

```bash
python -m scripts.prepare_protected_storage \
  --dataset-dir /protected/inkbridge/pilot-001 \
  --pilot-id pilot-shadow-001 \
  --storage-id storage-shadow-001
```

The command removes all group/other access, creates an empty private
`protected_lifecycle_events.jsonl`, and writes a deliberately invalid, read-only
`protected_storage_control.json` with `status: pending`. It never manufactures approval.

An accountable owner must temporarily make the control writable, replace every placeholder, bind
the actual root hash and current UID/GID, truthfully approve the asserted controls for at most 90
days, and make the file read-only again. The repository template is also deliberately invalid:
`configs/datasets/protected_storage_control.template.json`.

Verify the boundary before intake:

```bash
python -m scripts.verify_protected_storage \
  --contract /protected/inkbridge/pilot-001/pilot_intake.json \
  --dataset-dir /protected/inkbridge/pilot-001
```

The pilot intake audit now carries the exact storage-control SHA-256, asserted-control SHA-256,
hashed execution principal, local-IAM result, lifecycle ledger SHA/head, and aggregate lifecycle
counts. The frozen manifest, inference authorization, generated execution attestation, and final
aggregate report remain bound to the storage-control and ledger-head hashes.

The short-lived inference authorization must name the same accountable owner and operator as the
approved storage control. The execution attestation must name that same operator; role drift fails
closed before protected output is accepted.

## 2. Record consent withdrawal and deletion evidence

Subject identifiers must not enter the ledger. The consent system supplies a keyed, non-reversible
`subject_token_sha256`; `scope_sha256` binds the exact protected object inventory. Each deletion
event also binds an opaque internal evidence reference and SHA-256 of the provider/backup deletion
evidence.

The only accepted state sequence is:

1. `consent_withdrawal_requested` — accountable owner role;
2. `primary_deletion_verified` — approved operator role;
3. `backup_deletion_verified` — approved operator role;
4. `withdrawal_closed` — contract-bound deletion-verification owner role.

Example:

```bash
python -m scripts.record_protected_lifecycle_event \
  --contract /protected/inkbridge/pilot-001/pilot_intake.json \
  --dataset-dir /protected/inkbridge/pilot-001 \
  --event-type consent_withdrawal_requested \
  --subject-token-sha256 <64-hex-keyed-token> \
  --scope-sha256 <64-hex-object-inventory-hash> \
  --actor-role pilot_privacy_owner \
  --evidence-reference withdrawal_request_001 \
  --evidence-sha256 <64-hex-evidence-hash>
```

Every fsynced record contains the asserted occurrence time, automatic ledger-recording time,
sequence number, previous-event hash, and its own canonical content hash. Duplicate requests/events,
reordered timestamps, changed scopes, wrong actor roles, closure before both deletion proofs,
future timestamps, and chain tampering are rejected. A closure recorded after the approved deletion
deadline remains a breach even if its occurrence time was backdated. Recording is refused in CI.

An open withdrawal immediately makes intake, frozen-manifest verification, inference, and
evaluation fail closed. A late deletion may still be recorded so remediation is not blocked, but
the historical deadline breach remains visible and prevents future pilot processing.

The hash chain is tamper-evident, not storage-level append-only by itself. Export ledger-head hashes
and provider deletion/audit records to an independently controlled immutable/WORM audit system for
production evidence.

## 3. Shadow-pilot preflight

Before loading a model, run:

```bash
python -m scripts.preflight_protected_shadow_pilot \
  --contract /protected/inkbridge/pilot-001/pilot_intake.json \
  --dataset-dir /protected/inkbridge/pilot-001 \
  --manifest-dir /protected/inkbridge/pilot-001/frozen_evaluation \
  --authorization /protected/inkbridge/pilot-001/protected_inference_authorization.json \
  --model-dir /protected/inkbridge/models/trocr-v1
```

The preflight reports only aggregate pass/blocker codes. It validates the complete intake/freeze
and storage lifecycle chain, the sealed local safetensors artifact, and the short-lived inference
authorization without loading the model or running inference. A non-ready result exits with code
2; operators must resolve the external evidence rather than bypassing the gate.
