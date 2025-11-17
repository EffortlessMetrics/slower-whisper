Feature: Enriching transcripts with audio features
  As a user of slower-whisper
  I want to enrich transcripts with prosody and emotion data
  So that text-only models can understand acoustic features

  Background:
    Given a clean test project directory

  Scenario: Enrich transcript with prosody features
    Given a transcribed file "speech.wav" exists
    When I enrich the transcript with prosody enabled
    Then the transcript has audio_state for all segments
    And each audio_state contains prosody data
    And each audio_state has extraction_status

  Scenario: Enrich transcript with emotion features
    Given a transcribed file "emotional.wav" exists
    When I enrich the transcript with emotion enabled
    Then the transcript has audio_state for all segments
    And each audio_state contains emotion data
    And the emotion data includes dimensional values

  Scenario: Enrich transcript with full features
    Given a transcribed file "complete.wav" exists
    When I enrich the transcript with prosody and emotion enabled
    Then the transcript has audio_state for all segments
    And each audio_state contains prosody data
    And each audio_state contains emotion data
    And each audio_state has a text rendering

  Scenario: Enrich entire directory
    Given transcribed files exist:
      | filename    |
      | file1.wav   |
      | file2.wav   |
      | file3.wav   |
    When I enrich the directory with prosody enabled
    Then all transcripts have audio_state data
    And each transcript has extraction_status

  Scenario: Enrich single transcript object
    Given a transcript object for "sample.wav"
    And the corresponding audio file exists
    When I enrich the transcript object with prosody enabled
    Then the enriched transcript has audio_state
    And the prosody features include pitch, energy, and rate

  Scenario: Skip already enriched transcripts
    Given a transcribed file "enriched.wav" exists
    And the transcript is already enriched
    When I enrich the directory with skip_existing enabled
    Then the transcript for "enriched.wav" is unchanged

  Scenario: Handle missing audio files gracefully
    Given a transcript JSON for "missing.wav" exists
    But the audio file "missing.wav" does not exist
    When I enrich the directory with prosody enabled
    Then the enrichment skips "missing.wav"
    And other files are enriched successfully

  Scenario: Audio state includes rendering
    Given a transcribed file "rendered.wav" exists
    When I enrich the transcript with prosody enabled
    Then each audio_state has a rendering field
    And the rendering is a human-readable text description
    And the rendering matches the format "[audio: ...]"

  Scenario: Prosody features are speaker-relative
    Given a transcribed file "speaker.wav" exists
    When I enrich the transcript with prosody enabled
    Then the prosody features use speaker-relative baselines
    And the baseline is computed from sampled segments
