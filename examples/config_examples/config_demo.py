#!/usr/bin/env python3
"""
Configuration Examples for slower-whisper.

This script demonstrates all configuration methods and precedence rules.
It shows how to load configs from files, environment variables, and CLI args.

Usage:
    # Show default configuration
    python config_demo.py

    # Load from config file
    python config_demo.py --demo file

    # Load from environment variables
    python config_demo.py --demo env

    # Show precedence example
    python config_demo.py --demo precedence
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transcription import EnrichmentConfig, TranscriptionConfig


def demo_defaults():
    """Show default configuration values."""
    print("=" * 80)
    print("DEFAULT CONFIGURATION")
    print("=" * 80)

    trans_config = TranscriptionConfig()
    print("\nTranscriptionConfig defaults:")
    print(f"  model: {trans_config.model}")
    print(f"  device: {trans_config.device}")
    print(f"  compute_type: {trans_config.compute_type}")
    print(f"  language: {trans_config.language}")
    print(f"  task: {trans_config.task}")
    print(f"  skip_existing_json: {trans_config.skip_existing_json}")
    print(f"  vad_min_silence_ms: {trans_config.vad_min_silence_ms}")
    print(f"  beam_size: {trans_config.beam_size}")

    enrich_config = EnrichmentConfig()
    print("\nEnrichmentConfig defaults:")
    print(f"  skip_existing: {enrich_config.skip_existing}")
    print(f"  enable_prosody: {enrich_config.enable_prosody}")
    print(f"  enable_emotion: {enrich_config.enable_emotion}")
    print(f"  enable_categorical_emotion: {enrich_config.enable_categorical_emotion}")
    print(f"  device: {enrich_config.device}")
    print(f"  dimensional_model_name: {enrich_config.dimensional_model_name}")
    print(f"  categorical_model_name: {enrich_config.categorical_model_name}")


def demo_from_file():
    """Show loading configuration from JSON files."""
    print("=" * 80)
    print("LOADING FROM CONFIG FILES")
    print("=" * 80)

    config_dir = Path(__file__).parent

    # Load transcription config
    trans_file = config_dir / "transcription_production.json"
    if trans_file.exists():
        trans_config = TranscriptionConfig.from_file(trans_file)
        print(f"\nLoaded from {trans_file.name}:")
        print(f"  model: {trans_config.model}")
        print(f"  device: {trans_config.device}")
        print(f"  language: {trans_config.language}")
        print(f"  vad_min_silence_ms: {trans_config.vad_min_silence_ms}")
        print(f"  beam_size: {trans_config.beam_size}")
    else:
        print(f"\n⚠ Config file not found: {trans_file}")

    # Load enrichment config
    enrich_file = config_dir / "enrichment_full.json"
    if enrich_file.exists():
        enrich_config = EnrichmentConfig.from_file(enrich_file)
        print(f"\nLoaded from {enrich_file.name}:")
        print(f"  enable_prosody: {enrich_config.enable_prosody}")
        print(f"  enable_emotion: {enrich_config.enable_emotion}")
        print(f"  enable_categorical_emotion: {enrich_config.enable_categorical_emotion}")
        print(f"  device: {enrich_config.device}")
    else:
        print(f"\n⚠ Config file not found: {enrich_file}")


def demo_from_env():
    """Show loading configuration from environment variables."""
    print("=" * 80)
    print("LOADING FROM ENVIRONMENT VARIABLES")
    print("=" * 80)

    # Set example environment variables
    os.environ["SLOWER_WHISPER_MODEL"] = "medium"
    os.environ["SLOWER_WHISPER_DEVICE"] = "cpu"
    os.environ["SLOWER_WHISPER_LANGUAGE"] = "es"
    os.environ["SLOWER_WHISPER_BEAM_SIZE"] = "8"

    os.environ["SLOWER_WHISPER_ENRICH_ENABLE_PROSODY"] = "true"
    os.environ["SLOWER_WHISPER_ENRICH_ENABLE_EMOTION"] = "false"
    os.environ["SLOWER_WHISPER_ENRICH_DEVICE"] = "cuda"

    print("\nSet environment variables:")
    print("  SLOWER_WHISPER_MODEL=medium")
    print("  SLOWER_WHISPER_DEVICE=cpu")
    print("  SLOWER_WHISPER_LANGUAGE=es")
    print("  SLOWER_WHISPER_BEAM_SIZE=8")
    print("  SLOWER_WHISPER_ENRICH_ENABLE_PROSODY=true")
    print("  SLOWER_WHISPER_ENRICH_ENABLE_EMOTION=false")
    print("  SLOWER_WHISPER_ENRICH_DEVICE=cuda")

    # Load from environment
    trans_config = TranscriptionConfig.from_env()
    print("\nTranscriptionConfig from environment:")
    print(f"  model: {trans_config.model}")
    print(f"  device: {trans_config.device}")
    print(f"  language: {trans_config.language}")
    print(f"  beam_size: {trans_config.beam_size}")

    enrich_config = EnrichmentConfig.from_env()
    print("\nEnrichmentConfig from environment:")
    print(f"  enable_prosody: {enrich_config.enable_prosody}")
    print(f"  enable_emotion: {enrich_config.enable_emotion}")
    print(f"  device: {enrich_config.device}")

    # Clean up
    for key in list(os.environ.keys()):
        if key.startswith("SLOWER_WHISPER"):
            del os.environ[key]


def demo_precedence():
    """Show configuration precedence in action."""
    print("=" * 80)
    print("CONFIGURATION PRECEDENCE DEMONSTRATION")
    print("=" * 80)

    print("\nPrecedence order (highest to lowest):")
    print("1. CLI flags (simulated with direct construction)")
    print("2. Config file")
    print("3. Environment variables")
    print("4. Defaults")

    print("\n" + "-" * 80)
    print("Scenario: Layering defaults → env → file → CLI")
    print("-" * 80)

    # Layer 1: Defaults
    print("\n1. Start with defaults:")
    config = TranscriptionConfig()
    print(f"   model={config.model}, device={config.device}, language={config.language}")

    # Layer 2: Environment
    print("\n2. Apply environment variables:")
    os.environ["SLOWER_WHISPER_DEVICE"] = "cpu"
    os.environ["SLOWER_WHISPER_BEAM_SIZE"] = "8"
    env_config = TranscriptionConfig.from_env()
    print("   Set: SLOWER_WHISPER_DEVICE=cpu, SLOWER_WHISPER_BEAM_SIZE=8")
    print(
        f"   Result: model={env_config.model}, device={env_config.device}, beam_size={env_config.beam_size}"
    )

    # Layer 3: Config file (simulated)
    print("\n3. Apply config file (simulated):")
    print('   File contains: {"model": "large-v3", "device": "cuda", "language": "en"}')
    file_config = TranscriptionConfig(model="large-v3", device="cuda", language="en")
    print(
        f"   Result: model={file_config.model}, device={file_config.device}, language={file_config.language}"
    )
    print("   Note: device=cuda from file overrides device=cpu from environment")

    # Layer 4: CLI flags (simulated)
    print("\n4. Apply CLI flags (simulated):")
    print("   CLI flag: --model base")
    final_config = TranscriptionConfig(
        model="base",  # CLI override
        device="cuda",  # From file
        language="en",  # From file
        beam_size=8,  # From env
    )
    print(
        f"   Final: model={final_config.model}, device={final_config.device}, language={final_config.language}, beam_size={final_config.beam_size}"
    )
    print("   ✓ CLI flag --model overrides file's model=large-v3")
    print("   ✓ File's device=cuda is used (no CLI override)")
    print("   ✓ Env's beam_size=8 is used (not in file or CLI)")

    # Clean up
    for key in list(os.environ.keys()):
        if key.startswith("SLOWER_WHISPER"):
            del os.environ[key]


def demo_creating_configs():
    """Show how to create and save custom configurations."""
    print("=" * 80)
    print("CREATING CUSTOM CONFIGURATIONS")
    print("=" * 80)

    # Create a custom transcription config
    print("\n1. Creating a custom transcription config:")
    custom_trans = TranscriptionConfig(
        model="medium",
        device="cuda",
        language="fr",
        beam_size=8,
        vad_min_silence_ms=300,
    )

    trans_dict = {
        "model": custom_trans.model,
        "device": custom_trans.device,
        "compute_type": custom_trans.compute_type,
        "language": custom_trans.language,
        "task": custom_trans.task,
        "skip_existing_json": custom_trans.skip_existing_json,
        "vad_min_silence_ms": custom_trans.vad_min_silence_ms,
        "beam_size": custom_trans.beam_size,
    }

    print("\nTranscriptionConfig for French with medium model:")
    print(json.dumps(trans_dict, indent=2))

    # Create a custom enrichment config
    print("\n2. Creating a custom enrichment config:")
    custom_enrich = EnrichmentConfig(
        enable_prosody=True,
        enable_emotion=True,
        enable_categorical_emotion=False,
        device="cuda",
    )

    enrich_dict = {
        "skip_existing": custom_enrich.skip_existing,
        "enable_prosody": custom_enrich.enable_prosody,
        "enable_emotion": custom_enrich.enable_emotion,
        "enable_categorical_emotion": custom_enrich.enable_categorical_emotion,
        "device": custom_enrich.device,
        "dimensional_model_name": custom_enrich.dimensional_model_name,
        "categorical_model_name": custom_enrich.categorical_model_name,
    }

    print("\nEnrichmentConfig with prosody and dimensional emotion:")
    print(json.dumps(enrich_dict, indent=2))

    print("\n3. Save to file (example):")
    print('   with open("custom_config.json", "w") as f:')
    print("       json.dump(config_dict, f, indent=2)")


def main():
    """Run configuration demonstrations."""
    import argparse

    parser = argparse.ArgumentParser(description="Configuration examples for slower-whisper")
    parser.add_argument(
        "--demo",
        choices=["defaults", "file", "env", "precedence", "create", "all"],
        default="all",
        help="Which demo to run",
    )

    args = parser.parse_args()

    if args.demo == "all":
        demo_defaults()
        print("\n\n")
        demo_from_file()
        print("\n\n")
        demo_from_env()
        print("\n\n")
        demo_precedence()
        print("\n\n")
        demo_creating_configs()
    elif args.demo == "defaults":
        demo_defaults()
    elif args.demo == "file":
        demo_from_file()
    elif args.demo == "env":
        demo_from_env()
    elif args.demo == "precedence":
        demo_precedence()
    elif args.demo == "create":
        demo_creating_configs()

    print("\n" + "=" * 80)
    print("For more information, see examples/config_examples/README.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
