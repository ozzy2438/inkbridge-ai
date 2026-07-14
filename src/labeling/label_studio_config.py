"""Label Studio Configuration for InkBridge AI.

Sets up Label Studio for handwriting annotation:
- OCR transcription task template
- Pre-annotation from model predictions
- Active learning integration
- Quality metrics tracking
"""

LABEL_STUDIO_CONFIG_XML = """
<View>
  <Header value="InkBridge AI - Handwriting Transcription" />
  
  <View style="display: flex; gap: 20px;">
    <!-- Image panel -->
    <View style="flex: 1;">
      <Image name="image" value="$image" zoom="true" rotateControl="true" />
    </View>
    
    <!-- Annotation panel -->
    <View style="flex: 1;">
      <Header value="Transcription" size="4" />
      <TextArea name="transcription" toName="image" 
               rows="6" editable="true" 
               placeholder="Type the transcription here..." />
      
      <Header value="Region Type" size="4" />
      <Choices name="region_type" toName="image" choice="single">
        <Choice value="student_handwriting" />
        <Choice value="printed_text" />
        <Choice value="crossed_out" />
        <Choice value="margin_insertion" />
        <Choice value="teacher_annotation" />
        <Choice value="unreadable" />
      </Choices>
      
      <Header value="Quality Flags" size="4" />
      <Choices name="quality_flags" toName="image" choice="multiple">
        <Choice value="blur" />
        <Choice value="faint_pencil" />
        <Choice value="overlapping_lines" />
        <Choice value="mixed_languages" />
        <Choice value="mathematical_notation" />
      </Choices>
      
      <Header value="Confidence" size="4" />
      <Rating name="annotator_confidence" toName="image" maxRating="5" />
    </View>
  </View>
  
  <!-- Bounding box annotations -->
  <RectangleLabels name="bbox" toName="image">
    <Label value="text_line" background="#2196F3" />
    <Label value="crossed_out" background="#F44336" />
    <Label value="insertion" background="#4CAF50" />
    <Label value="printed" background="#9C27B0" />
  </RectangleLabels>
</View>
"""


ANNOTATION_GUIDELINES = """
# InkBridge AI — Annotation Guidelines

## Core Principles
1. Transcribe EXACTLY what is written (do not correct grammar or spelling)
2. Mark crossed-out text with [CROSSED: original text]
3. Mark insertions with [INSERT: inserted text]
4. If unreadable, mark as [UNREADABLE]
5. Preserve original punctuation

## Handling Ambiguity
- If a letter could be uppercase or lowercase, choose the most likely
- If spacing is unclear, use your best judgment
- If punctuation is ambiguous, mark it in notes

## Cross-outs
- Still transcribe the crossed-out text
- Mark it clearly so the system knows it was deleted
- If crossed-out text is unreadable, use [CROSSED: UNREADABLE]

## Insertions
- Text added between lines or in margins
- Note the intended reading order
- Mark where in the flow it should appear

## Printed vs Handwritten
- Separate printed questions from student answers
- Mark headers and page numbers as printed

## When Two Annotators Disagree
- Flag for adjudication
- Record both interpretations
- Senior annotator makes final decision
"""


def get_label_studio_project_config() -> dict:
    """Get Label Studio project configuration."""
    return {
        "title": "InkBridge AI - Handwriting Transcription",
        "description": "Annotate and correct handwriting transcriptions",
        "label_config": LABEL_STUDIO_CONFIG_XML,
        "expert_instruction": ANNOTATION_GUIDELINES,
        "show_instruction": True,
        "enable_empty_annotation": False,
        "show_skip_button": True,
        "show_annotation_history": True,
    }
