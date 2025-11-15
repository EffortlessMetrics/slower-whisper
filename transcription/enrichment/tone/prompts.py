"""Prompts for LLM-based tone analysis."""

TONE_LABELS_DESC = """
- neutral: Calm, matter-of-fact delivery, informational
- positive: Enthusiastic, happy, excited, encouraging, optimistic
- negative: Frustrated, angry, disappointed, critical, sad
- questioning: Curious, probing, seeking clarification, inquisitive
- uncertain: Hesitant, doubtful, tentative, unsure
- emphatic: Strong emphasis, passionate, assertive, forceful
"""


def build_tone_analysis_prompt(segments_text: str, include_context: bool = False) -> str:
    """
    Build the prompt for tone analysis.

    Args:
        segments_text: Formatted segment text to analyze
        include_context: Whether context was included

    Returns:
        Formatted prompt string
    """
    context_note = ""
    if include_context:
        context_note = "\n(Context segments are provided for understanding, but only analyze the segments marked with [ANALYZE].)"

    return f"""You are analyzing the emotional tone of speech segments from a transcript.

For each segment below, classify the tone using ONE of these labels:
{TONE_LABELS_DESC}

Also provide a confidence score from 0.0 to 1.0 indicating how certain you are of the classification.
{context_note}

Segments to analyze:
{segments_text}

Respond with ONLY a JSON array, one object per segment in order, with this exact format:
[
  {{"segment_id": 0, "tone": "neutral", "confidence": 0.85}},
  {{"segment_id": 1, "tone": "positive", "confidence": 0.92}},
  ...
]

Do not include any other text or explanation, just the JSON array."""


def format_segments_for_analysis(segments, start_idx: int, end_idx: int,
                                  context_before=None, context_after=None) -> str:
    """
    Format segments for the prompt.

    Args:
        segments: List of Segment objects to analyze
        start_idx: Start index in the full segment list
        end_idx: End index in the full segment list
        context_before: Optional list of context segments before
        context_after: Optional list of context segments after

    Returns:
        Formatted segment text
    """
    lines = []

    # Add context before
    if context_before:
        for seg in context_before:
            lines.append(f"[CONTEXT] Segment {seg.id}: \"{seg.text}\"")
        lines.append("")

    # Add segments to analyze
    for i, seg in enumerate(segments):
        lines.append(f"[ANALYZE] Segment {seg.id}: \"{seg.text}\"")

    # Add context after
    if context_after:
        lines.append("")
        for seg in context_after:
            lines.append(f"[CONTEXT] Segment {seg.id}: \"{seg.text}\"")

    return "\n".join(lines)
