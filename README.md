# MMS Spoken LID

## Description

Spoken language identification (LID) over fixed-length windows of an audio track, using the language identification models of [Massively Multilingual Speech (MMS)](https://huggingface.co/facebook/mms-lid-4017). The app reads the audio of an `AudioDocument` or `VideoDocument` in back-to-back windows of `windowMs` through `mmif.utils.audio_document_helper`, and runs MMS on each window.

The `model` parameter picks one of the MMS LID models, which differ only in the number of languages they label: 126, 256, 512, 1024, 2048 or 4017 (the default).

MMS gives one label per window: its most probable language. MMS labels are ISO 639-3 codes, and the app writes them as BCP 47 language tags: the two-letter ISO 639-1 code where the same language has one (`eng` becomes `en`, `spa` becomes `es`), else the ISO 639-3 code (`esu` for Central Alaskan Yup'ik). Consecutive windows with the same label become one `TimeFrame`, with `start` and `end` in milliseconds. Its `scores` list holds, in time order, the MMS probability of that label in each of the merged windows, so a low-confidence window inside a long segment stays visible. With `tfMerge=false`, each window becomes its own `TimeFrame`. The view's `TimeFrame` metadata lists all labels of the model as `labelset`.

### Keeping only some labels

With `labelMap`, the app keeps only the labels you name, for example to tell English from everything else. Each entry is `IN:OUT`, which renames the label `IN` to `OUT`, or just `IN`, which keeps it as it is; several labels can map to the same `OUT`. All other labels become `-`.

The label of a window then comes from the MMS probabilities for all of its languages in that window. The probabilities of the labels mapped to the same `OUT` are added up, and the window gets the `OUT` label with the largest total if that total is at least `minLabelScore` (default 0.5); otherwise it gets `-`. The score of a window is the total of its label; for `-`, the total of all labels that are not mapped. For example, with `labelMap=en`, a window is `en` when MMS gives English at least 0.5, and `-` otherwise. A lower `minLabelScore` labels more windows `en`, a higher one fewer.

`OUT` labels are scored separately, so with `labelMap=en es` English and Spanish each have to reach `minLabelScore` on their own. To count several labels together, map them to the same `OUT`, as in `labelMap=en:target es:target`.

The probabilities are the softmax of the MMS classifier over its labels. They are not calibrated, so a total of 0.6 does not mean the label is right 60% of the time, and the best `minLabelScore` for a collection is best found on labeled data. The `labelset` lists the `OUT` labels and `-`.

## User instruction

General user instructions for CLAMS apps are available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

### System requirements

The app runs on CPU or, when available, a CUDA GPU. Each MMS LID model is a 1-billion-parameter wav2vec2 model; its weights (about 3.9 GB) are downloaded on first use to the Hugging Face cache (`$HF_HOME`).

The MMS models are licensed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), for non-commercial use only. The app code is licensed under Apache 2.0.

### Configurable runtime parameters

For the full list of parameters, see the app metadata in the [CLAMS App Directory](https://apps.clams.ai) or [`metadata.py`](metadata.py).

## Development

Run the tests with `pytest` from the repository root. The tests replace the model with a stub, so no model download is needed.
