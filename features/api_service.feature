Feature: API Service
  As a slower-whisper API consumer
  I want the REST API to be available and functional
  So that I can integrate transcription into my applications

  Background:
    Given the API service is running

  @api @smoke
  Scenario: Health check endpoint responds
    When I send a GET request to "/health"
    Then the response status code should be 200
    And the response should contain "status"
    And the response should contain "service"
    And the response should contain "schema_version"

  @api @smoke
  Scenario: OpenAPI documentation is available
    When I send a GET request to "/docs"
    Then the response status code should be 200
    And the response content type should be "text/html"

  @api @smoke
  Scenario: OpenAPI JSON schema is available
    When I send a GET request to "/openapi.json"
    Then the response status code should be 200
    And the response content type should be "application/json"
    And the response should contain "openapi"
    And the response should contain "paths"

  @api @functional @requires_ffmpeg
  Scenario: Transcribe endpoint accepts audio and returns transcript
    Given I have a sample audio file "test_audio.wav"
    When I POST the audio to "/transcribe" with model "base" and device "cpu"
    Then the response status code should be 200
    And the response should contain "segments"
    And the response should contain "schema_version"
    And the schema version should be 2

  @api @functional @requires_enrich
  Scenario: Enrich endpoint adds audio features to transcript
    Given I have a sample transcript file "test_transcript.json"
    And I have a sample audio file "test_audio.wav"
    When I POST both files to "/enrich" with prosody enabled
    Then the response status code should be 200
    And the response should contain "segments"
    And at least one segment should have "audio_state"
