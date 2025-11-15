"""
LLM Integration Examples

This package provides tools and examples for integrating audio-enriched
transcripts with Large Language Models.

Modules:
- prompt_builder: Build LLM-ready prompts from enriched transcripts
- analysis_queries: Pre-built query templates for common analysis tasks
- llm_integration_demo: Full workflow with API integration
- comparison_demo: Compare analysis with vs without audio enrichment

Quick Start:
    from llm_integration import prompt_builder, analysis_queries

    # Build a prompt
    builder = prompt_builder.TranscriptPromptBuilder("transcript.json")
    prompt = builder.build_analysis_prompt("Find moments of escalation")

    # Use pre-built queries
    templates = analysis_queries.QueryTemplates("transcript.json")
    escalation_prompt = templates.find_escalation_moments()
"""

__version__ = "1.0.0"
__all__ = ["prompt_builder", "analysis_queries", "llm_integration_demo", "comparison_demo"]
