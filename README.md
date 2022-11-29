# Music Gen

This project aims to train Convolutional AutoEncoder(s) based on various types of music. The decoder will then be used to generate music from the latent features.

## Feature Structure

Notes:

- I add a feature for "modernness" to the latent representation, which is not encoded but should assist the decoder.

## Installation

- Libraries required:
  - https://github.com/ai-unit/midi-orchestra (lib/midi-orchestra-master/)
    - `pip install -r requirements.txt`
