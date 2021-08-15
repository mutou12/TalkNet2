# feature extraction related
sample_rate = 22050  # sampling frequency
fmax = 8000.0  # maximum frequency
fmin = 0.0  # minimum frequency
n_mels = 80  # number of mel basis
n_fft = 1024  # number of fft points
hop_length = 256  # number of shift points
win_length = 1024  # window length
num_mels = 80
min_level_db = -100
ref_level_db = 20
bits = 9  # bit depth of signal
mu_law = True  # Recommended to suppress noise if using raw bits in hp.voc_mode below
peak_norm = False  # Normalise to the peak of each wav file


epochs = 400
batch_size = 16

idim = 358
f0_mean = 247.6103
f0_std = 57.9735