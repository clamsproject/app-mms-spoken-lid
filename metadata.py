"""
The purpose of this file is to define the metadata of the app with minimal imports.

DO NOT CHANGE the name of the file
"""
from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from mmif import DocumentTypes, AnnotationTypes

# MMS LID models by name, with the Hugging Face commit of their weights. All
# are the same 1B wav2vec2 model with heads for different numbers of
# languages.
MODEL_REVISIONS = {
    'facebook/mms-lid-126': '53da6e311f3ce48f324fe335924216187e3109a4',
    'facebook/mms-lid-256': 'edc73fd00996e671dfc59d16436a29b12b10588a',
    'facebook/mms-lid-512': 'e537f7f8d3d7f1058a14d4f73dbffc69ec52fc8e',
    'facebook/mms-lid-1024': '6216408c37833201358b509224b019312577dd70',
    'facebook/mms-lid-2048': '44f27d589b0c9351ccf3e900fbf0c6aaca06e24c',
    'facebook/mms-lid-4017': '6adb04d61a52e5989b0e65f0d59d4755d81a94e3',
}
DEFAULT_MODEL = 'facebook/mms-lid-4017'


def appmetadata() -> AppMetadata:
    metadata = AppMetadata(
        name="MMS Spoken LID",
        description=(
            "Spoken language identification over fixed-length windows of "
            "an audio track, using the MMS language identification models. "
            "Consecutive windows with the same language can be merged into "
            "one TimeFrame."),
        app_license="Apache 2.0",
        identifier="mms-spoken-lid",
        url="https://github.com/clamsproject/app-mms-spoken-lid",
        analyzer_versions=dict(MODEL_REVISIONS),
        analyzer_license="CC-BY-NC-4.0",
        # measured on an RTX A6000 with 10 s windows: about 4.4 GB reserved
        # by torch at batch size 1 (5.9 GB at 8), plus the CUDA context
        est_gpu_mem_min=5000,
        est_gpu_mem_typ=5000,
    )
    metadata.add_input_oneof(DocumentTypes.AudioDocument,
                             DocumentTypes.VideoDocument)
    out_tf = metadata.add_output(AnnotationTypes.TimeFrame,
                                 timeUnit='milliseconds', document='*',
                                 labelset='*', label='*', scores='*')
    out_tf.add_description(
        'Each TimeFrame covers consecutive windows that MMS labels with the '
        'language `label`. Language labels are canonicalized to BCP 47 '
        'tags, and `labelMap` can reduce them; see README.md in the app '
        'repository for details. `labelset` lists all labels the view can '
        'have. `scores` lists, in time order, the MMS probability of '
        '`label` in each of those windows, one float in [0, 1] per window. '
        'Window i starts at `start` + i * `windowMs`; the last window ends '
        'at `end` and can be shorter. The probabilities are not '
        'calibrated.')

    metadata.add_parameter(
        name='model', type='string', default=DEFAULT_MODEL,
        choices=list(MODEL_REVISIONS),
        description='MMS LID model, by the number of languages it labels.')
    metadata.add_parameter(
        name='windowMs', type='integer', default=10000,
        description='Window length in milliseconds.')
    metadata.add_parameter(
        name='minLastWindowMs', type='integer', default=0,
        description='Shortest final window to keep, in milliseconds. 0 '
                    'keeps any final window.')
    metadata.add_parameter(
        name='tfMerge', type='boolean', default=True,
        description='Join consecutive windows with the same label into one '
                    'TimeFrame. When false, each window becomes its own '
                    'TimeFrame, with one score.')
    metadata.add_parameter(
        name='labelMap', type='string', default=[], multivalued=True,
        description='Labels to keep, as "IN:OUT" (rename IN to OUT) or "IN" '
                    '(keep IN as it is). Other labels become "-". Empty '
                    'keeps every label of the model. See README.md in the '
                    'app repository for how the label is chosen.')
    metadata.add_parameter(
        name='minLabelScore', type='number', default=0.5,
        description='Minimum score for a label from labelMap to be kept; a '
                    'window whose best mapped label scores less becomes '
                    '"-". Only applies with labelMap.')
    metadata.add_parameter(
        name='batchSize', type='integer', default=1,
        description='Number of windows passed to the model at once.')
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
