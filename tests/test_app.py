import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
import torch
from mmif import AnnotationTypes, Document, DocumentTypes, Mmif

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402

RATE = 16000
SECONDS = 3.5
# ISO 639-3 labels of the stub model, in id order
CODES = ['eng', 'spa', 'fra', 'esu', 'smo']
# most probable language of each 1000 ms window of the test audio
TOPS = ['eng', 'eng', 'fra', 'eng']


def top_probs(i):
    """Probabilities with TOPS[i] on top at 0.6, 0.7 or 0.8."""
    top = TOPS[i % len(TOPS)]
    p = 0.6 + 0.1 * (i % 3)
    rest = (1 - p) / (len(CODES) - 1)
    return [p if code == top else rest for code in CODES]


class FakeModel:
    """Stands in for ``Wav2Vec2ForSequenceClassification``: records batch
    sizes and input lengths, and returns logits whose softmax is
    ``probs_for`` of each window, in window order."""

    def __init__(self, probs_for=top_probs):
        self.probs_for = probs_for
        self.config = SimpleNamespace(id2label=dict(enumerate(CODES)))
        self.batch_sizes = []
        self.seen = 0

    def eval(self):
        return self

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, input_values, attention_mask):
        assert input_values.shape == attention_mask.shape
        self.batch_sizes.append(len(input_values))
        probs = torch.tensor([self.probs_for(self.seen + i)
                              for i in range(len(input_values))])
        self.seen += len(input_values)
        return SimpleNamespace(logits=torch.log(probs.clamp_min(1e-12)))


class FakeExtractor:
    """Stands in for ``Wav2Vec2FeatureExtractor``: pads to the longest
    window and returns a mask."""

    def __call__(self, windows, sampling_rate, padding, return_attention_mask,
                 return_tensors):
        assert sampling_rate == RATE and padding and return_attention_mask
        longest = max(len(w) for w in windows)
        values = torch.zeros(len(windows), longest)
        mask = torch.zeros(len(windows), longest, dtype=torch.long)
        for row, samples in enumerate(windows):
            values[row, :len(samples)] = torch.from_numpy(samples)
            mask[row, :len(samples)] = 1
        return _Batch(input_values=values, attention_mask=mask)


class _Batch(dict):
    def to(self, device):
        return self


def use_model(monkeypatch, fake):
    monkeypatch.setattr(app._MmsLid, 'load', classmethod(
        lambda cls, model_id: cls(fake, FakeExtractor())))
    return fake


@pytest.fixture
def model(monkeypatch):
    return use_model(monkeypatch, FakeModel())


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / 'tone.wav'
    t = np.arange(int(RATE * SECONDS)) / RATE
    samples = (np.sin(2 * np.pi * 440 * t) * 10000).astype('int16')[None, :]
    with av.open(str(path), 'w') as out:
        stream = out.add_stream('pcm_s16le', rate=RATE, layout='mono')
        frame = av.AudioFrame.from_ndarray(samples, format='s16',
                                           layout='mono')
        frame.sample_rate = RATE
        frame.pts = 0
        frame.time_base = Fraction(1, RATE)
        out.mux(stream.encode(frame))
        out.mux(stream.encode(None))
    return path


def run(audio_path, **params):
    mmif = Mmif(validate=False)
    mmif.add_document(Document({
        '@type': DocumentTypes.AudioDocument,
        'properties': {'id': 'd1', 'mime': 'audio/wav',
                       'location': f'file://{audio_path}'}}))
    params = {k: [str(x) for x in v] if isinstance(v, list) else [str(v)]
              for k, v in params.items()}
    return Mmif(app.get_app().annotate(mmif, **params))


def timeframes(out):
    view = out.get_view_contains(AnnotationTypes.TimeFrame)
    tfs = list(view.get_annotations(AnnotationTypes.TimeFrame))
    return view, [(tf.get_property('start'), tf.get_property('end'),
                   tf.get_property('label'),
                   [round(s, 3) for s in tf.get_property('scores')])
                  for tf in tfs]


def test_canonical_codes():
    assert app._canonical('eng') == 'en'
    assert app._canonical('spa') == 'es'
    assert app._canonical('smo') == 'sm'
    # no ISO 639-1 code: the ISO 639-3 code stays
    assert app._canonical('esu') == 'esu'
    assert app._canonical('cmn') == 'cmn'
    # not an ISO 639-3 code: left as it is
    assert app._canonical('xx-unknown') == 'xx-unknown'


def test_segments():
    windows = [(0, 10, 'en', .9), (10, 20, 'en', .8), (20, 25, 'fr', .7)]
    assert list(app._segments(windows)) == [
        (0, 20, 'en', [.9, .8]), (20, 25, 'fr', [.7])]
    assert list(app._segments(windows, merge=False)) == [
        (0, 10, 'en', [.9]), (10, 20, 'en', [.8]), (20, 25, 'fr', [.7])]
    # windows that do not touch stay apart
    assert list(app._segments([(0, 10, 'en', .9), (15, 20, 'en', .8)])) == [
        (0, 10, 'en', [.9]), (15, 20, 'en', [.8])]


def test_same_language_windows_are_merged(model, audio_path):
    view, tfs = timeframes(run(audio_path, windowMs=1000))
    assert tfs == [(0, 2000, 'en', [0.6, 0.7]), (2000, 3000, 'fr', [0.8]),
                   (3000, 3500, 'en', [0.6])]
    contain = view.metadata.contains[AnnotationTypes.TimeFrame]
    assert contain['document'] == 'd1'
    assert contain['timeUnit'] == 'milliseconds'
    assert contain['labelset'] == ['en', 'es', 'fr', 'esu', 'sm']


def test_merge_off_gives_one_timeframe_per_window(model, audio_path):
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False))
    assert [(s, e) for s, e, _, _ in tfs] == [
        (0, 1000), (1000, 2000), (2000, 3000), (3000, 3500)]
    assert all(len(scores) == 1 for _, _, _, scores in tfs)


def test_short_last_window_is_dropped(model, audio_path):
    _, tfs = timeframes(run(audio_path, windowMs=1000, minLastWindowMs=600))
    assert tfs[-1][:3] == (2000, 3000, 'fr')


def test_window_too_short_for_the_model_is_skipped(model, audio_path):
    # 3500 ms in 1160 ms windows leaves a 20 ms tail, shorter than 25 ms
    _, tfs = timeframes(run(audio_path, windowMs=1160, tfMerge=False))
    assert tfs[-1][1] == 3480


def test_batches_follow_batch_size(model, audio_path):
    _, tfs = timeframes(run(audio_path, windowMs=1000, batchSize=3))
    assert model.batch_sizes == [3, 1]
    assert len(tfs) == 3


def test_model_parameter_picks_the_model(monkeypatch, audio_path):
    loaded = []
    monkeypatch.setattr(app._MmsLid, 'load', classmethod(
        lambda cls, model_id: loaded.append(model_id)
        or cls(FakeModel(), FakeExtractor())))
    run(audio_path, windowMs=1000, model='facebook/mms-lid-126')
    assert loaded == ['facebook/mms-lid-126']


# probabilities over CODES (eng, spa, fra, esu, smo) for each 1000 ms window
PROBS = [[.6, .1, .1, .1, .1],
         [.3, .3, .4, 0, 0]]


@pytest.fixture
def pooled(monkeypatch):
    return use_model(monkeypatch,
                     FakeModel(lambda i: PROBS[i % len(PROBS)]))


def test_parse_label_map():
    labelset = ['en', 'es', 'fr']
    assert app._parse_label_map(['en', 'es:spanish'], labelset) == {
        'en': 'en', 'es': 'spanish'}
    with pytest.raises(ValueError, match='"xx" is not a label'):
        app._parse_label_map(['xx:en'], labelset)


def test_label_map_pools_probabilities(pooled, audio_path):
    view, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False,
                               labelMap='en'))
    # window 2: en 0.3 against 0.7 for all other labels
    assert [(label, scores[0]) for _, _, label, scores in tfs] == [
        ('en', 0.6), ('-', 0.7), ('en', 0.6), ('-', 0.7)]
    contain = view.metadata.contains[AnnotationTypes.TimeFrame]
    assert contain['labelset'] == ['en', '-']


def test_labels_mapped_together_add_up(pooled, audio_path):
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False,
                            labelMap=['en:target', 'es:target']))
    assert [(label, scores[0]) for _, _, label, scores in tfs] == [
        ('target', 0.7), ('target', 0.6), ('target', 0.7), ('target', 0.6)]


def test_label_map_takes_bcp47_tags(pooled, audio_path):
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False,
                            labelMap=['esu', 'en'], minLabelScore=0.05))
    assert [label for _, _, label, _ in tfs] == ['en', 'en', 'en', 'en']
    with pytest.raises(ValueError, match='"eng" is not a label'):
        run(audio_path, windowMs=1000, labelMap='eng')


def test_min_label_score_sets_the_bar(pooled, audio_path):
    # window 2: en 0.3, below the default 0.5 but above 0.25
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False,
                            labelMap='en', minLabelScore=0.25))
    assert [(label, scores[0]) for _, _, label, scores in tfs] == [
        ('en', 0.6), ('en', 0.3), ('en', 0.6), ('en', 0.3)]
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False,
                            labelMap='en', minLabelScore=0.7))
    assert [label for _, _, label, _ in tfs] == ['-'] * 4


def test_separate_labels_each_need_the_score(pooled, audio_path):
    # window 2: en 0.3 and es 0.3, each below 0.5
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False,
                            labelMap=['en', 'es']))
    assert [(label, scores[0]) for _, _, label, scores in tfs] == [
        ('en', 0.6), ('-', 0.4), ('en', 0.6), ('-', 0.4)]
