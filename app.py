"""
MMS Spoken LID: language identification over fixed-length windows of an
audio track with the MMS language identification models, with the windows
read through the SDK's audio document helper. Consecutive windows with the
same language can be merged into one TimeFrame.
"""
import argparse
import functools
import logging
from itertools import chain, islice

import pycountry
import torch
from clams import ClamsApp, Restifier
from metadata import MODEL_REVISIONS
from mmif import AnnotationTypes, DocumentTypes, Mmif
from mmif.utils import audio_document_helper as adh
from transformers import AutoFeatureExtractor
from transformers import Wav2Vec2ForSequenceClassification

_SAMPLE_RATE = 16000
# the receptive field of the wav2vec2 feature encoder, 25 ms at 16 kHz; a
# shorter window gives no frame
_MIN_SAMPLES = 400
# with a labelMap, the label of a window whose labels are mostly unmapped
_NEGATIVE_LABEL = '-'


@functools.lru_cache(maxsize=None)
def _canonical(code):
    """The BCP 47 tag of an ISO 639-3 code: the ISO 639-1 code of the same
    language where there is one, else the ISO 639-3 code itself."""
    language = pycountry.languages.get(alpha_3=code)
    return getattr(language, 'alpha_2', code) if language else code


class _MmsLid:
    """An MMS LID model, run on in-memory windows."""

    def __init__(self, model, feature_extractor):
        """
        :param model: loaded ``Wav2Vec2ForSequenceClassification``
        :param feature_extractor: its ``Wav2Vec2FeatureExtractor``
        """
        self._model = model.eval()
        self._extractor = feature_extractor
        self._device = next(model.parameters()).device
        self._labels = [_canonical(model.config.id2label[i])
                        for i in range(len(model.config.id2label))]

    @classmethod
    def load(cls, model_id):
        revision = MODEL_REVISIONS[model_id]
        extractor = AutoFeatureExtractor.from_pretrained(model_id,
                                                         revision=revision)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_id, revision=revision)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return cls(model.to(device), extractor)

    @property
    def labels(self):
        """All labels of the model, as BCP 47 tags."""
        return list(self._labels)

    @torch.no_grad()
    def detect(self, windows, label_map=None, min_score=0.5):
        """
        :param windows: mono ``float32`` sample arrays at 16 kHz
        :param label_map: ``{IN: OUT}``; without it, the label is the most
            probable language
        :param min_score: with ``label_map``, the minimum total of the best
            ``OUT`` label for it to be kept
        :returns: ``(label, score)`` per window, or ``None`` for a window too
            short for the model (25 ms)
        """
        results = [None] * len(windows)
        kept = [i for i, samples in enumerate(windows)
                if len(samples) >= _MIN_SAMPLES]
        if not kept:
            return results
        inputs = self._extractor(
            [windows[i] for i in kept], sampling_rate=_SAMPLE_RATE,
            padding=True, return_attention_mask=True, return_tensors='pt')
        logits = self._model(**inputs.to(self._device)).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu()
        for i, row in zip(kept, probs):
            if label_map:
                results[i] = _pooled_label(
                    dict(zip(self._labels, row.tolist())), label_map,
                    min_score)
            else:
                best = int(row.argmax())
                results[i] = (self._labels[best], float(row[best]))
        return results


def _pooled_label(probs, label_map, min_score):
    """
    :param probs: ``{label: probability}`` over all labels of the model
    :returns: ``(label, probability)``. The probability of each ``OUT``
        label is the sum over the labels mapped to it. The most probable
        ``OUT`` label wins if its probability is at least ``min_score``;
        otherwise the label is ``_NEGATIVE_LABEL``, with the total of the
        labels that are not mapped.
    """
    pooled = {}
    for label, p in probs.items():
        target = label_map.get(label, _NEGATIVE_LABEL)
        pooled[target] = pooled.get(target, 0.0) + p
    unmapped = pooled.pop(_NEGATIVE_LABEL, 0.0)
    # empty when every label is mapped to the negative label
    best = max(pooled, key=pooled.get) if pooled else None
    if best is not None and pooled[best] >= min_score:
        return best, pooled[best]
    return _NEGATIVE_LABEL, unmapped


def _batched(iterable, size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch


def _segments(labeled_windows, merge=True):
    """
    :param labeled_windows: ``(start_ms, end_ms, label, score)`` per window,
        in time order
    :param merge: join consecutive windows with the same label, when each
        starts where the previous one ends
    :returns: ``(start_ms, end_ms, label, scores)`` per segment, with one
        score per window in time order
    """
    segment = None
    for start, end, label, score in labeled_windows:
        if (merge and segment is not None and segment[2] == label
                and segment[1] == start):
            segment[1] = end
            segment[3].append(score)
            continue
        if segment is not None:
            yield tuple(segment)
        segment = [start, end, label, [score]]
    if segment is not None:
        yield tuple(segment)


def _parse_label_map(entries, labelset):
    """
    :param entries: ``IN:OUT`` strings; an entry without a colon keeps the
        label as it is
    :returns: ``{IN: OUT}``
    :raises ValueError: if an ``IN`` label is not a label of the model
    """
    label_map = {}
    for entry in entries:
        source, _, target = (part.strip() for part in entry.partition(':'))
        if source not in labelset:
            raise ValueError(f'labelMap: "{source}" is not a label of the '
                             f'model; see the labelset in the app metadata')
        label_map[source] = target or source
    return label_map


def _mapped_labelset(label_map):
    labels = list(dict.fromkeys(label_map.values()))
    return labels + [_NEGATIVE_LABEL] * (_NEGATIVE_LABEL not in labels)


class MmsSpokenLid(ClamsApp):

    def __init__(self):
        super().__init__()
        self._models = {}

    def _appmetadata(self):
        pass

    def _get_model(self, model_id):
        if model_id not in self._models:
            self.logger.debug(f'Loading {model_id}@'
                              f'{MODEL_REVISIONS[model_id]}')
            self._models[model_id] = _MmsLid.load(model_id)
        return self._models[model_id]

    def _label_windows(self, model, windows, batch_size, label_map,
                       min_score):
        """
        :param windows: ``(samples, start_ms, end_ms)`` per window
        :returns: ``(start_ms, end_ms, label, score)`` per window; a window
            too short for the model is left out
        """
        for batch in _batched(windows, batch_size):
            detected = model.detect([samples for samples, _, _ in batch],
                                    label_map, min_score)
            for (_, start_ms, end_ms), result in zip(batch, detected):
                if result is None:
                    self.logger.debug(f'Window {start_ms}-{end_ms} ms is too '
                                      f'short for MMS, skipped')
                    continue
                label, score = result
                yield start_ms, end_ms, label, score

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        model = self._get_model(parameters['model'])
        labels = model.labels
        label_map = _parse_label_map(parameters['labelMap'], labels)
        labelset = _mapped_labelset(label_map) if label_map else labels
        docs = chain(mmif.get_documents_by_type(DocumentTypes.AudioDocument),
                     mmif.get_documents_by_type(DocumentTypes.VideoDocument))
        for doc in docs:
            view = mmif.new_view()
            self.sign_view(view, parameters)
            view.new_contain(AnnotationTypes.TimeFrame, document=doc.id,
                             timeUnit='milliseconds', labelset=labelset)
            windows = adh.iter_audio_windows(
                doc, parameters['windowMs'], sample_rate=_SAMPLE_RATE,
                min_last_window_ms=parameters['minLastWindowMs'])
            labeled = self._label_windows(model, windows,
                                          parameters['batchSize'],
                                          label_map,
                                          parameters['minLabelScore'])
            for start_ms, end_ms, label, scores in _segments(
                    labeled, parameters['tfMerge']):
                view.new_annotation(AnnotationTypes.TimeFrame,
                                    start=start_ms, end=end_ms,
                                    label=label, scores=scores)
        return mmif


def get_app():
    return MmsSpokenLid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000",
                        help="set port to listen")
    parser.add_argument("--production", action="store_true",
                        help="run gunicorn server")
    parsed_args = parser.parse_args()

    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
