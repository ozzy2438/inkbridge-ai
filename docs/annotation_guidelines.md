# InkBridge AI — Annotation Guidelines v2

## Purpose and boundary

These rules define how protected handwriting samples become gold-candidate transcriptions. A gold
candidate receives two blind human passes and, when needed, independent adjudication. It is not a
locked gold set until writer-isolated splits are frozen and approved.

Model suggestions are disabled during gold creation to prevent anchoring bias. Model-assisted
pre-annotation may be used later for operational labelling, but those records must stay separate
from the locked evaluation set.

## Roles

- **Annotator 1 and Annotator 2:** transcribe independently and cannot see each other's work.
- **Adjudicator:** resolves disagreements and must be different from both annotators.
- **Educator/domain reviewer:** decides ambiguous classroom conventions and approves the guideline.
- **Data owner:** controls access, consent withdrawal, retention, and deletion evidence.
- **ML owner:** maintains schemas and metrics but cannot silently change a gold reference.

Annotator and writer identifiers are opaque pseudonyms. Names, emails, student IDs, dates of birth,
consent records, and approval documents never belong in the normalized dataset.

## Transcription rules

### Fidelity to the original

- Transcribe exactly what is visible; do not correct spelling or grammar.
- Preserve capitalization, punctuation, spacing that changes meaning, and paragraph boundaries.
- Do not expand abbreviations or infer a word from the essay topic.
- Do not copy a printed prompt into the student-handwriting transcript.

### Crop boundaries

- Include the complete target line and every mark that belongs to it.
- Exclude adjacent lines, student names/headers, teacher marks, and printed questions.
- Mark `crop_boundary_disagreement` when the correct boundary is uncertain.
- A crop is not eligible until both annotators confirm `crop_reviewed=true`.

### Cross-outs

- Preserve readable crossed-out text as `[CROSSED: text here]`.
- Use `[CROSSED: UNREADABLE]` when the underlying text cannot be recovered.
- Do not replace a crossed-out word with a later correction; record both in reading order.

### Insertions and reading order

- Place an insertion at its intended reading position when a caret or clear marker identifies it.
- Encode it as `[INSERT@line3: text]`, using the target line number.
- If intent is ambiguous, preserve visual order and escalate as `reading_order_disagreement`.

### Unreadable and partially readable text

- Use `[UNREADABLE]` only after zoom/contrast inspection cannot resolve a token.
- Use `[PARTIAL: wh_t]` for a partly legible token; one underscore represents each unknown
  character where count is discernible.
- Never guess a likely word to improve fluency.

### Printed, student, and teacher content

- `printed_text`: questions, headers, ruled-form labels, or machine-printed content.
- `student_handwriting`: the target student's answer.
- `teacher_annotation`: ticks, scores, comments, or corrections from an educator.
- `diagram_non_text`: drawings or diagrams without a transcription target.

Mixed authorship is escalated as `region_type_disagreement`; annotators must not infer authorship
from ink colour alone.

### Special cases

| Situation | Required action |
|---|---|
| Mathematical notation | Preserve in plain text, for example `x^2 + 3x = 0` |
| Arrow or symbol | Use a concise token such as `[ARROW_RIGHT]` |
| Blank crop/page | Reject from line gold data; record as a document-quality case |
| Multiple languages | Transcribe as visible and flag `multilingual_review` |
| Ambiguous punctuation | Transcribe the visible mark; escalate if annotators disagree |
| Student identity in crop | Stop annotation and return the sample for de-identification |

## Independent review protocol

1. The data owner assigns opaque sample IDs and removes direct/visual identifiers and image
   metadata.
2. Annotator 1 and Annotator 2 work blind, without model pre-annotations.
3. Each pass stores a SHA-256 of its exact UTF-8 transcription and time spent.
4. Exact agreement is recorded as `agreed`; both pass hashes and final hash must match.
5. Disagreement is recorded as `adjudicated`, with an independent adjudicator, controlled reason,
   final-transcript hash, and adjudication time.
6. The final hash must match the exact text in `labels.csv`.
7. The protected-pilot gate verifies every sample before producing a PII-free audit.

Allowed adjudication reasons are:

- `transcription_disagreement`
- `uncertain_character`
- `reading_order_disagreement`
- `crop_boundary_disagreement`
- `region_type_disagreement`

The adjudicator records a concise rationale in the protected annotation system. The public audit
contains only counts, rates, and hashes—not rationale text, student content, or worker identities.

## Quality and operations metrics

- Exact agreement rate
- Adjudication rate
- Total review hours and samples per review hour
- Correction time per page during a later shadow pilot
- Rework rate after educator review
- Agreement and error rate by failure slice

No fixed agreement target can make a dataset gold by itself. Low agreement triggers guideline or
sample-quality review; it must not be hidden by adjudication.

## Failure-category tagging

| Category | Description |
|---|---|
| `faint_pencil` | Writing too light to read reliably |
| `joined_cursive` | Connected letters are difficult to segment |
| `crossed_out` | Struck-through text is present |
| `margin_insertion` | Text appears between lines or in margins |
| `touching_lines` | Adjacent lines overlap |
| `mixed_content` | Printed and handwritten content coexist |
| `heavy_blur` | Motion or defocus blur obscures writing |
| `perspective` | Camera angle causes severe distortion |
| `teacher_annotation` | Teacher marks overlap student writing |
| `missing_page_edge` | Capture truncates relevant content |

## Promotion boundary

A package that passes annotation review is only `gold_candidate_ready`. Promotion requires a
writer-isolated, versioned manifest frozen before model selection; a locked test set must never be
used for training, threshold tuning, prompt selection, or manual reference changes without a new
dataset version.
