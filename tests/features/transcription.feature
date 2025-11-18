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

  # v1.1 Speaker diarization scenarios
  Scenario: Transcripts have nullable speaker fields by default
    Given a project with a mono WAV file named "conversation.wav"
    When I transcribe the project with default settings
    Then a transcript JSON exists for "conversation.wav"
    And all segments have a "speaker" field
    And all segment "speaker" values are null (diarization disabled)

  Scenario: Schema v2 supports speakers and turns arrays
    Given a project with a mono WAV file named "meeting.wav"
    When I transcribe the project with default settings
    Then the transcript JSON has schema version 2
    And the transcript may contain a "speakers" array
    And the transcript may contain a "turns" array
    And if "speakers" array exists, each speaker has required fields

  @requires_diarization
  Scenario: Speaker diarization with pyannote.audio backend
    Given a project with a mono WAV file named "test.wav"
    And the HF_TOKEN environment variable is set
    When I transcribe the project with diarization enabled
    Then the transcription completes successfully
    And the transcript JSON has meta.diarization.status "success"
    And meta.diarization.backend is "pyannote.audio"
    And meta.diarization.requested is true
    And the "speakers" array is populated with detected speakers
    And the "turns" array contains turn structure

  @requires_diarization
  Scenario: Diarization on 2-speaker synthetic fixture
    Given a project with the synthetic 2-speaker fixture
    And the HF_TOKEN environment variable is set
    When I transcribe the project with diarization enabled (min=2, max=2)
    Then the transcript JSON has meta.diarization.status "success"
    And meta.diarization.requested is true
    And meta.diarization.num_speakers equals 2
    And the "speakers" array has exactly 2 speakers
    And the "turns" array has at least 2 turns
    And speaker turns alternate between the two detected speakers

  Scenario: Diarization gracefully handles missing dependencies
    Given a project with a mono WAV file named "test.wav"
    And pyannote.audio is not installed
    When I transcribe the project with diarization enabled
    Then the transcription completes successfully
    And the transcript JSON has meta.diarization.status "failed"
    And meta.diarization.requested is true
    And meta.diarization.error_type is "missing_dependency"
    And the "speakers" field is null
    And the "turns" field is null

  Scenario: Diarization gracefully handles missing HF token
    Given a project with a mono WAV file named "test.wav"
    And the HF_TOKEN environment variable is not set
    When I transcribe the project with diarization enabled
    Then the transcription completes successfully
    And the transcript JSON has meta.diarization.status "failed"
    And meta.diarization.requested is true
    And meta.diarization.error_type is "auth"
    And the "speakers" field is null
    And the "turns" field is null
