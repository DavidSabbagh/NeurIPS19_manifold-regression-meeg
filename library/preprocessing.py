import os.path as op

import mne
from autoreject import get_rejection_threshold

# from ..config_drago import path_maxfilter_info
path_maxfilter_info = '/storage/local/camcan/maxfilter'

# epoching params
duration = 30.  # length of epochs (in s)
overlap = 8.  # shift of overlapping epochs (in s)
n_fft = 8192  # length of hamming windows (in #samples)
n_overlap = 4096  # overlap of overlapping hamm win (in #samples)

# spectra params
fmin = 0.  # min freq of spectra
fmax = 150.  # max freq of spectra
fbands = [(0.1, 1.5),  # low
          (1.5, 4.0),  # delta
          (4.0, 8.0),  # theta
          (8.0, 15.0),  # alpha
          (15.0, 26.0),  # beta_low
          (26.0, 35.0),  # beta_high
          (35.0, 50.0),  # gamma_low
          (50.0, 74.0),  # gamma_mid
          (76.0, 120.0)]  # gamma_high


def get_subject(file_raw):
    return file_raw.split('/')[-3]


# cleaning functions
def _get_global_reject_epochs(raw):
    duration = 3.
    events = mne.make_fixed_length_events(
        raw, id=3000, start=0, duration=duration)

    epochs = mne.Epochs(
        raw, events, event_id=3000, tmin=0, tmax=duration, proj=False,
        baseline=None, reject=None)
    epochs.load_data()
    epochs.pick_types(meg=True)
    reject = get_rejection_threshold(epochs, decim=1)
    return reject


def _get_global_reject_ssp(raw):
    if 'eog' in raw:
        eog_epochs = mne.preprocessing.create_eog_epochs(raw)
    else:
        eog_epochs = []
    if len(eog_epochs) >= 5:
        reject_eog = get_rejection_threshold(eog_epochs, decim=8)
        del reject_eog['eog']  # we don't want to reject eog based on eog
    else:
        reject_eog = None

    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
    # we will always have an ECG as long as there are magnetometers
    if len(ecg_epochs) >= 5:
        reject_ecg = get_rejection_threshold(ecg_epochs, decim=8)
        # here we want the eog
    else:
        reject_ecg = None

    if reject_eog is None and reject_ecg is not None:
        reject_eog = {k: v for k, v in reject_ecg.items() if k != 'eog'}
    return reject_eog, reject_ecg


def _compute_add_ssp_exg(raw):
    reject_eog, reject_ecg = _get_global_reject_ssp(raw)
    if 'eog' in raw:
        proj_eog, _ = mne.preprocessing.compute_proj_eog(
            raw, average=True, reject=reject_eog, n_mag=1, n_grad=1, n_eeg=1)
    else:
        proj_eog = None
    if proj_eog is not None:
        raw.add_proj(proj_eog)

    proj_ecg, _ = mne.preprocessing.compute_proj_ecg(
        raw, average=True, reject=reject_ecg, n_mag=1, n_grad=1, n_eeg=1)
    if proj_ecg is not None:
        raw.add_proj(proj_ecg)


def parse_bad_channels(sss_log):
    """Parse bad channels from sss_log."""
    with open(sss_log) as fid:
        bad_lines = {l for l in fid.readlines() if 'Static bad' in l}
    bad_channels = list()
    for line in bad_lines:
        chans = line.split(':')[1].strip(' \n').split(' ')
        for cc in chans:
            ch_name = 'MEG%01d' % int(cc)
            if ch_name not in bad_channels:
                bad_channels.append(ch_name)
    return bad_channels


def _parse_bads(subject, kind):
    sss_log = op.join(
        path_maxfilter_info, subject,
        kind, "mf2pt2_{kind}_raw.log".format(kind=kind))

    try:
        bads = parse_bad_channels(sss_log)
    except Exception as err:
        print(err)
        bads = []
    # first 100 channels ommit the 0.
    bads = [''.join(['MEG', '0', bb.split('MEG')[-1]])
            if len(bb) < 7 else bb for bb in bads]
    return bads


def _run_maxfilter(raw, subject, kind):
    bads = _parse_bads(subject, kind)
    raw.info['bads'] = bads
    cal = op.join(path_maxfilter_info, 'sss_params', 'sss_cal.dat')
    ctc = op.join(path_maxfilter_info, 'sss_params', 'ct_sparse.fif')
    raw = mne.preprocessing.maxwell_filter(
        raw, calibration=cal,
        cross_talk=ctc,
        st_duration=10.,
        st_correlation=.98,
        destination=None,
        coord_frame='head')
    return raw


def clean_raw(raw, subject):
    mne.channels.fix_mag_coil_types(raw.info)
    raw = _run_maxfilter(raw, subject, 'rest')
    _compute_add_ssp_exg(raw)
    reject = _get_global_reject_epochs(raw)
    return raw, reject


def minclean_raw(raw, subject):
    mne.channels.fix_mag_coil_types(raw.info)
    # bads = _parse_bads(subject, 'rest')
    # raw.info['bads'] = bads
    raw.add_proj([], remove_existing=True)
    # raw.interpolate_bads(reset_bads=False)
    reject = None
    return raw, reject
