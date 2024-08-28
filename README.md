This code contains a swift implementation of the [Denoiser model](https://github.com/facebookresearch/denoiser/tree/main) based on [Demucs](https://github.com/facebookresearch/demucs/tree/main).

The code is completely OOC, and aims to be a helper and a base for people wanting to integrate it.

## Project structure

_> `extensions` folder : contains helpfull extensions, not every one is used in the project but they may be helpful.

_> `denoiser.swift` : is a naive implementation using MPS Graphs. It follows the exact same model structure as the python implementation, and can be use with AVAudioPCMBuffer.

_> `denoiser_realtime.swift` : following the naive implementation, and some swift memory mysteries, the weights took so much memory it was impossible to run the denoiser in real time. This file contains some changes made to try improving this, but honestly it still needs some modifications for me.

_> `export_weight.py` : basic python code to export bin files of the differents layers of the denoiser python model. Some example code is present, but you can just use this as a base.

## Improvements

If anyone wants to talk about this implementation, feel free to contact me or to propose modifications.