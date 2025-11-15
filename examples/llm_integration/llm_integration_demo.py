#!/usr/bin/env python3
"""
Complete LLM Integration Demo

This script demonstrates a full end-to-end workflow:
1. Load an enriched transcript
2. Build an LLM-ready prompt
3. Send to an LLM API (OpenAI, Anthropic, etc.)
4. Parse and display the results

The demo includes placeholders for API keys and shows best practices
for integrating enriched transcripts with various LLM providers.

Supported LLMs:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Local models via OpenAI-compatible APIs

Usage:
    # Set your API key as an environment variable
    export OPENAI_API_KEY="your-key-here"
    # or
    export ANTHROPIC_API_KEY="your-key-here"

    # Run the demo
    python llm_integration_demo.py enriched_transcript.json --provider openai
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis_queries import QueryTemplates
from prompt_builder import TranscriptPromptBuilder


@dataclass
class LLMConfig:
    """Configuration for LLM API calls."""

    provider: Literal["openai", "anthropic", "local"]
    model: str
    api_key: str | None = None
    api_base: str | None = None  # For local/custom endpoints
    temperature: float = 0.7
    max_tokens: int = 2000


class LLMIntegration:
    """
    Integration layer for various LLM providers.

    Handles API calls, response parsing, and error handling.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM integration.

        Args:
            config: LLM configuration including provider and API key
        """
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Validate configuration and check for required API keys."""
        if self.config.provider == "openai":
            if not self.config.api_key:
                self.config.api_key = os.getenv("OPENAI_API_KEY")
            if not self.config.api_key:
                print("WARNING: OPENAI_API_KEY not set. API calls will fail.")
                print("Set it with: export OPENAI_API_KEY='your-key-here'")

        elif self.config.provider == "anthropic":
            if not self.config.api_key:
                self.config.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.config.api_key:
                print("WARNING: ANTHROPIC_API_KEY not set. API calls will fail.")
                print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")

    def query(self, prompt: str, system_message: str | None = None) -> dict[str, Any]:
        """
        Send a query to the LLM.

        Args:
            prompt: The prompt text
            system_message: Optional system message (provider-dependent)

        Returns:
            Dictionary with 'response', 'usage', and 'metadata'
        """
        if self.config.provider == "openai":
            return self._query_openai(prompt, system_message)
        elif self.config.provider == "anthropic":
            return self._query_anthropic(prompt, system_message)
        elif self.config.provider == "local":
            return self._query_local(prompt, system_message)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def _query_openai(self, prompt: str, system_message: str | None = None) -> dict[str, Any]:
        """Query OpenAI API."""
        try:
            import openai
        except ImportError:
            return {
                "response": "[ERROR: openai package not installed. Install with: pip install openai]",
                "usage": {},
                "metadata": {"error": "missing_package"},
            }

        if not self.config.api_key:
            return {
                "response": "[ERROR: OPENAI_API_KEY not set]",
                "usage": {},
                "metadata": {"error": "missing_api_key"},
            }

        try:
            client = openai.OpenAI(api_key=self.config.api_key)

            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            return {
                "response": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "metadata": {
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason,
                },
            }

        except Exception as e:
            return {"response": f"[ERROR: {str(e)}]", "usage": {}, "metadata": {"error": str(e)}}

    def _query_anthropic(self, prompt: str, system_message: str | None = None) -> dict[str, Any]:
        """Query Anthropic API."""
        try:
            import anthropic
        except ImportError:
            return {
                "response": "[ERROR: anthropic package not installed. Install with: pip install anthropic]",
                "usage": {},
                "metadata": {"error": "missing_package"},
            }

        if not self.config.api_key:
            return {
                "response": "[ERROR: ANTHROPIC_API_KEY not set]",
                "usage": {},
                "metadata": {"error": "missing_api_key"},
            }

        try:
            client = anthropic.Anthropic(api_key=self.config.api_key)

            kwargs = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }

            if system_message:
                kwargs["system"] = system_message

            response = client.messages.create(**kwargs)

            return {
                "response": response.content[0].text,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                "metadata": {"model": response.model, "stop_reason": response.stop_reason},
            }

        except Exception as e:
            return {"response": f"[ERROR: {str(e)}]", "usage": {}, "metadata": {"error": str(e)}}

    def _query_local(self, prompt: str, system_message: str | None = None) -> dict[str, Any]:
        """Query local/custom OpenAI-compatible API."""
        try:
            import openai
        except ImportError:
            return {
                "response": "[ERROR: openai package not installed]",
                "usage": {},
                "metadata": {"error": "missing_package"},
            }

        try:
            client = openai.OpenAI(
                api_key=self.config.api_key or "not-needed",
                base_url=self.config.api_base or "http://localhost:8000/v1",
            )

            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            return {
                "response": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                },
                "metadata": {
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason,
                },
            }

        except Exception as e:
            return {"response": f"[ERROR: {str(e)}]", "usage": {}, "metadata": {"error": str(e)}}


def run_demo_workflow(
    transcript_path: Path, provider: str = "openai", model: str | None = None, dry_run: bool = False
):
    """
    Run a complete demo workflow.

    Args:
        transcript_path: Path to enriched JSON transcript
        provider: LLM provider (openai, anthropic, local)
        model: Model name (provider-specific)
        dry_run: If True, only show prompts without calling API
    """
    print("=" * 80)
    print("LLM INTEGRATION DEMO - Full Workflow")
    print("=" * 80)
    print()

    # Step 1: Load transcript
    print("STEP 1: Loading Enriched Transcript")
    print("-" * 80)
    print(f"Source: {transcript_path}")

    try:
        builder = TranscriptPromptBuilder(transcript_path)
        templates = QueryTemplates(transcript_path)
        print(f"✓ Loaded transcript: {builder.transcript.file_name}")
        print(f"  - Language: {builder.transcript.language}")
        print(f"  - Segments: {len(builder.transcript.segments)}")
        print()
    except Exception as e:
        print(f"✗ Error loading transcript: {e}")
        return

    # Step 2: Configure LLM
    print("STEP 2: Configuring LLM")
    print("-" * 80)

    if not model:
        model = {
            "openai": "gpt-4",
            "anthropic": "claude-3-5-sonnet-20241022",
            "local": "local-model",
        }.get(provider, "gpt-4")

    config = LLMConfig(provider=provider, model=model, temperature=0.7, max_tokens=2000)

    print(f"Provider: {config.provider}")
    print(f"Model: {config.model}")
    print(f"Temperature: {config.temperature}")
    print(f"Max tokens: {config.max_tokens}")

    if dry_run:
        print("Mode: DRY RUN (prompts only, no API calls)")
    print()

    # Step 3: Generate prompts and query
    analyses = [
        ("Find Moments of Escalation", templates.find_escalation_moments()),
        ("Identify Uncertain Statements", templates.find_uncertain_statements()),
        ("Summarize Emotional Arc", templates.summarize_emotional_arc()),
    ]

    llm = LLMIntegration(config)

    for i, (title, prompt) in enumerate(analyses, 1):
        print(f"STEP {i + 2}: {title}")
        print("-" * 80)

        if dry_run:
            print("PROMPT (first 500 chars):")
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            print()
            continue

        print("Querying LLM...")
        start_time = time.time()

        result = llm.query(
            prompt=prompt,
            system_message="You are an expert at analyzing transcripts with audio enrichment data.",
        )

        elapsed = time.time() - start_time

        print(f"✓ Response received in {elapsed:.2f}s")
        print()

        if "error" not in result["metadata"]:
            print("RESPONSE:")
            print(result["response"])
            print()

            if result["usage"]:
                print("USAGE:")
                print(f"  Prompt tokens: {result['usage'].get('prompt_tokens', 'N/A')}")
                print(f"  Completion tokens: {result['usage'].get('completion_tokens', 'N/A')}")
                print(f"  Total tokens: {result['usage'].get('total_tokens', 'N/A')}")
        else:
            print("ERROR:")
            print(result["response"])

        print()
        print("=" * 80)
        print()

        # Rate limiting pause
        if not dry_run and i < len(analyses):
            print("Waiting 2s before next query...")
            time.sleep(2)


def demonstrate_api_integration():
    """Show example API integration patterns."""
    print("=" * 80)
    print("LLM API INTEGRATION EXAMPLES")
    print("=" * 80)
    print()

    print("EXAMPLE 1: Basic OpenAI Integration")
    print("-" * 80)
    print("""
import openai
from prompt_builder import TranscriptPromptBuilder

# Load transcript and build prompt
builder = TranscriptPromptBuilder("transcript.json")
prompt = builder.build_analysis_prompt("Find moments of escalation")

# Query OpenAI
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an expert at analyzing transcripts."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    max_tokens=2000
)

print(response.choices[0].message.content)
""")
    print()

    print("EXAMPLE 2: Anthropic Claude Integration")
    print("-" * 80)
    print("""
import anthropic
from analysis_queries import QueryTemplates

# Load transcript and generate query
templates = QueryTemplates("transcript.json")
prompt = templates.detect_sarcasm_contradiction()

# Query Anthropic
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2000,
    system="You are an expert at analyzing conversational tone and sarcasm.",
    messages=[{"role": "user", "content": prompt}]
)

print(response.content[0].text)
""")
    print()

    print("EXAMPLE 3: Batch Processing Multiple Transcripts")
    print("-" * 80)
    print("""
from pathlib import Path
from analysis_queries import QueryTemplates
from llm_integration_demo import LLMIntegration, LLMConfig

# Configure LLM
config = LLMConfig(provider="openai", model="gpt-4")
llm = LLMIntegration(config)

# Process all transcripts
results = []
for transcript_file in Path("transcripts/").glob("*.json"):
    templates = QueryTemplates(transcript_file)
    prompt = templates.summarize_emotional_arc()

    result = llm.query(prompt)
    results.append({
        "file": transcript_file.name,
        "analysis": result["response"],
        "tokens": result["usage"]["total_tokens"]
    })

# Save results
import json
with open("batch_analysis.json", "w") as f:
    json.dump(results, f, indent=2)
""")
    print()

    print("EXAMPLE 4: Streaming Responses (OpenAI)")
    print("-" * 80)
    print("""
import openai

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    stream=True
)

print("Streaming response: ", end="", flush=True)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
""")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM integration demo")
    parser.add_argument("transcript", nargs="?", type=Path, help="Path to enriched JSON transcript")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "local"],
        default="openai",
        help="LLM provider",
    )
    parser.add_argument("--model", help="Model name (provider-specific)")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts without calling API")
    parser.add_argument("--examples", action="store_true", help="Show API integration examples")

    args = parser.parse_args()

    if args.examples or not args.transcript:
        demonstrate_api_integration()
    else:
        run_demo_workflow(
            transcript_path=args.transcript,
            provider=args.provider,
            model=args.model,
            dry_run=args.dry_run,
        )
