Feature: Transcribing audio into transcripts
  As a user of slower-whisper
  I want to transcribe audio files
  So that I can get text transcripts in multiple formats

  Background:
    Given a clean test project directory

  Scenario: Transcribe a simple audio file
    Given a project with a mono WAV file named "hello.wav"
    When I transcribe the project with default settings
    Then a transcript JSON exists for "hello.wav"
    And the transcript contains at least one segment
    And the JSON file has schema version 2

  Scenario: Transcribe with custom model configuration
    Given a project with a mono WAV file named "test.wav"
    When I transcribe the project with model "base" and language "en"
    Then a transcript JSON exists for "test.wav"
    And the transcript language is "en"
    And the transcript metadata contains model "base"

  Scenario: Transcribe produces multiple output formats
    Given a project with a mono WAV file named "sample.wav"
    When I transcribe the project with default settings
    Then a transcript JSON exists for "sample.wav"
    And a transcript TXT exists for "sample.wav"
    And a transcript SRT exists for "sample.wav"

  Scenario: Transcribe single file directly
    Given an audio file "interview.wav"
    When I transcribe the single file with default settings
    Then the transcript has segments
    And the transcript file name is "interview.wav"

  Scenario: Skip existing transcripts when configured
    Given a project with a mono WAV file named "existing.wav"
    And the file "existing.wav" has already been transcribed
    When I transcribe the project with skip_existing_json enabled
    Then the transcript JSON for "existing.wav" is unchanged

  Scenario: Transcribe handles multiple files
    Given a project with audio files:
      | filename    |
      | audio1.wav  |
      | audio2.wav  |
      | audio3.wav  |
    When I transcribe the project with default settings
    Then transcript JSONs exist for all files
    And each transcript contains at least one segment
