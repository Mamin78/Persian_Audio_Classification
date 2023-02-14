import numpy as np
import librosa
import librosa.display


def noise(data):
    noise_amp = 0.04 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.70):
    return librosa.effects.time_stretch(data, rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=0.8):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


def higher_speed(data, speed_factor=1.25):
    return librosa.effects.time_stretch(data, speed_factor)


def lower_speed(data, speed_factor=0.75):
    return librosa.effects.time_stretch(data, speed_factor)


def extract_features(data):
    result = np.array([])
    mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    result = np.array(mfccs_processed)

    return result


def load_audio(path):
    data, sample_rate = librosa.load(path, duration=3, offset=0.5, res_type='kaiser_fast')
    return data, sample_rate


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen
    # above.
    data, sample_rate = load_audio(path)

    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    # noised
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))  # stacking vertically

    # stretched
    stretch_data = stretch(data)
    res3 = extract_features(stretch_data)
    result = np.vstack((result, res3))

    # shifted
    shift_data = shift(data)
    res4 = extract_features(shift_data)
    result = np.vstack((result, res4))

    # pitched
    pitch_data = pitch(data, sample_rate)
    res5 = extract_features(pitch_data)
    result = np.vstack((result, res5))

    # speed up
    higher_speed_data = higher_speed(data)
    res6 = extract_features(higher_speed_data)
    result = np.vstack((result, res6))

    # speed down
    lower_speed_data = higher_speed(data)
    res7 = extract_features(lower_speed_data)
    result = np.vstack((result, res7))

    return result
