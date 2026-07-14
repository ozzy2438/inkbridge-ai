# InkBridge AI — Annotation Guidelines

## Purpose
This document defines the standards for transcribing student handwriting samples in the InkBridge AI system.

## Core Rules

### 1. Fidelity to Original
- Transcribe EXACTLY what is written
- Do NOT correct spelling errors
- Do NOT correct grammar
- Preserve original capitalization
- Preserve original punctuation

### 2. Cross-outs
- Still transcribe crossed-out text
- Mark with tag: `[CROSSED: text here]`
- If unreadable under cross-out: `[CROSSED: UNREADABLE]`

### 3. Insertions
- Text added between lines or in margins
- Mark with: `[INSERT: text here]`
- Note position indicator: `[INSERT@line3: text]`

### 4. Unreadable Text
- If genuinely cannot be read: `[UNREADABLE]`
- If partially readable: `[PARTIAL: wh_t I c_n read]`
- Use underscore for unreadable characters

### 5. Reading Order
- Top to bottom, left to right (default)
- Follow numbered sections if present
- Insertions placed at intended position

### 6. Printed vs Handwritten
- Mark region type during annotation
- Printed questions: mark as `printed_text`
- Student answers: mark as `student_handwriting`
- Teacher marks: mark as `teacher_annotation`

### 7. Special Cases
| Situation | Action |
|-----------|--------|
| Mathematical notation | Transcribe in plain text: "x^2 + 3x = 0" |
| Drawings/diagrams | Mark as `diagram_non_text` |
| Arrows/symbols | Describe: "[ARROW pointing right]" |
| Blank page | Mark as blank, no transcription |
| Multiple languages | Transcribe as-is, flag for review |

## Quality Metrics

### Agreement Rate
- Target: >90% inter-annotator agreement
- Measured at character level

### Adjudication Rules
- If two annotators disagree by >10 characters: escalate
- Senior annotator makes final decision
- Decision and reasoning recorded

### Speed Targets
- Target: 4-6 pages per hour per annotator
- Quality over speed always
- Report if consistently below target

## Failure Category Tagging

When a model prediction is incorrect, tag the failure:

| Category | Description | Example |
|----------|-------------|---------|
| `faint_pencil` | Writing too light to read | Light HB pencil on white paper |
| `joined_cursive` | Letters connected, hard to segment | Flowing cursive writing |
| `crossed_out` | Struck-through text | Single/double line through words |
| `margin_insertion` | Text in margins/between lines | Caret insertions |
| `touching_lines` | Adjacent lines overlap | Cramped writing |
| `mixed_content` | Printed and handwritten together | Question + answer |
| `heavy_blur` | Camera blur | Motion or defocus blur |
| `perspective` | Severe angle | Phone photo from side |
