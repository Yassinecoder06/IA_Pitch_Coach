# Third-party notices

## iamspeaker

Parts of the speech-analysis design and the word-choice lexicon in `shared/language.json`
(hedging, vague and apology expressions; plain-English suggestions), the monotone-pitch
threshold, and the take-to-take comparison model (higher-is-better metric scores, coaching
notes classified as resolved / persisting / new) are adapted from
[iamspeaker](https://github.com/spark798/iamspeaker).

MIT License

Copyright (c) 2026 Seung Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## MediaPipe

Face and pose tracking use [MediaPipe Tasks Vision](https://github.com/google-ai-edge/mediapipe)
(Apache License 2.0). Its WASM runtime and the face/pose landmarker models are downloaded by
`web/scripts/fetch-mediapipe.mjs` and served locally.

## Other projects consulted

SpeechMax and speakforge-app were reviewed for feature ideas only. No code from them is
included: SpeechMax is "All rights reserved" and speakforge-app has no license.
