# A simple server for `OpenVoice`

This project provides an OpenAI compatible text-to-speech server for the 
OpenVoice project.

[OpenVoice](https://github.com/myshell-ai/OpenVoice/) is an Open Source project
of Text-to-speech model and tools.

## Features

- text-to-speech
- tone management
  - create tone by upload wave files
  - list tones
- openai-compatible '/v1/audio/speech' endpoint

## Install

You need to build the docker images.

```sh
cd src
docker build . -t openvoice-server
```

The Dockerfile requires the download of all related models, hence the resulting
docker image can run in an isolated environment.

Currently this project only supports CUDA.

## Run

To run the server, use the docker image you just built:

```sh
docker run --rm -p 18080:8080  --device=nvidia.com/gpu=all openvoice-server:latest
```

### Environment variables

- `DATABASE_URL`
  The server stores tone information in a SQLite database, you can control
  the position of the database by specifying the `DATABASE_URL` environment
  variable.

## Use the API

### GET `/tones`

Get a list of existing tones

### POST `/tones`

Create a tone by uploading a `.wav` file. 

- Request `Content-Type: multipart/form-data`

Fields in request:
- audiofile
- name
- desc
  
- Response `Content-Type: application/json`

See `examples/create-tone.sh` for the curl command.

### GET `/tones/{tone_id}`

Get the details of a tone by `tone_id`.

- Response `Content-Type: application/json`

### POST `/tones/{tone_id}/{dialect}`

Generate voice using the specified `tone_id` and `dialect`.

- Request `Content-Type: application/json`

Fields in request:
- text
- speed[optional]

- Response `Content-Type: audio/x-wav`

See `examples/generate_speech.sh` for the curl command

### GET `/cap`

Get a list of supported languages and dialects.

### POST `/v1/audio/speech`

The OpenAI compatible endpoint. 

See `examples/generate_speech_openai.sh` for the curl command.

Note:

1. use `dialect` as model in request, find dialects from the `/cap` endpoint.
1. use `tone_name` as voice in request, find tones from the `/tones` endpoint.


