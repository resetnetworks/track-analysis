"""
=============================================================================
Audio Fingerprinting Engine (Shazam-style Algorithm)

PURPOSE:
This script extracts unique "fingerprints" from audio files. It serves as the
core engine for audio identification and matching (e.g., copyright detection).

HOW IT WORKS:
1. Spectrogram: Converts audio into a frequency-over-time map.
2. Peak Detection: Finds the most prominent frequencies (constellation map) 
   that are highly resistant to background noise.
3. Hashing: Pairs these peaks based on time offsets and hashes them (SHA-1) 
   to create fast, searchable database entries.

USAGE WORKFLOW:
- Ingestion: Run this on your music catalogue and store hashes in a Database.
- Matching: Run this on an unknown audio snippet and query the DB for matches.
=============================================================================
"""
import essentia.standard as es
import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt

'''
todo: Use signal processing (STFT + hashing) -- needed to do
this is basic
'''

def plot(spec):
    plt.figure(figsize=(10, 6))

    # work on copy
    spec_vis = spec.copy()

    # normalize
    spec_vis = spec_vis / np.max(spec_vis)

    # clip extreme values
    spec_vis = np.clip(spec_vis, 0, np.percentile(spec_vis, 99))

    # threshold to highlight peaks
    threshold = 0.6 * np.max(spec_vis)
    spec_vis[spec_vis < threshold] = 0

    plt.imshow(spec_vis.T, aspect='auto', origin='lower')

    plt.title("Spectrogram (Enhanced Peaks)")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")

    plt.colorbar(label="Normalized Log Amplitude")
    plt.show()


'''working:
🎧 Convert to time
frame_duration = frameSize / sampleRate
Example:
2048 @ 44100 Hz → ~46 ms
4096 @ 44100 Hz → ~93 ms
👉 That"s the time chunk each FFT sees
'''


# Generate Spectrogram
def get_spectrogram(audio_path):
    loader = es.MonoLoader(filename=audio_path)
    audio = loader()

    window = es.Windowing(type='hann')
    spectrum = es.Spectrum()

    # frames = es.FrameGenerator(audio, frameSize=2048, hopSize=256)
    # frames = es.FrameGenerator(audio, frameSize=2048, hopSize=512)
    # frames = es.FrameGenerator(audio, frameSize=4096, hopSize=1024) # seems best for now.
    frames = es.FrameGenerator(audio, frameSize=4096, hopSize=512)

    print("FRAME", frames)

    spectrogram = []
 
    for frame in frames:
        w = window(frame)
        spec = spectrum(w)

        # 🔥 Convert to log scale (key fix)
        spec = np.log1p(spec)
        spectrogram.append(spec)

    spectrogram = np.array(spectrogram)
    # plot(spectrogram)
    return np.array(spectrogram)


# Peak Detection
def find_peaks(spectrogram, amp_min=0.1):
    peaks = []

    for t, frame in enumerate(spectrogram):
        for f, amp in enumerate(frame):
            if amp > amp_min:
                peaks.append((t, f))

    return peaks


# Fingerprint Hashing
import hashlib

def generate_hashes(peaks, fan_value=5):
    hashes = []

    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if i + j < len(peaks):
                t1, f1 = peaks[i]
                t2, f2 = peaks[i + j]

                dt = t2 - t1

                hash_input = f"{f1}|{f2}|{dt}"
                h = hashlib.sha1(hash_input.encode()).hexdigest()

                hashes.append((h, t1))

    return hashes


def save_json(file_name, data, data_type):
    if data_type == "peaks":
        folder = os.path.join("effective_fingerprint", "peaks")
    elif data_type == "hashes":
        folder = os.path.join("effective_fingerprint", "hashes")
    else:
        raise ValueError("Invalid data_type. Use 'peaks' or 'hashes'")

    os.makedirs(folder, exist_ok=True)

    output_file = os.path.join(folder, file_name)

    with open(output_file, "w") as f:
        json.dump(data, f)


# Provide the path to the audio file you want to process here
src_name = "new-audio/Yiruma, (이루마) - River Flows in You.mp3"
file_name = os.path.basename(src_name)
file_name1 = os.path.splitext(file_name)[0]

print("\n" + "="*50)
print("🎵 PROCESSING AUDIO FILE")
print("="*50)

print(f"File Name: {file_name1}")

time_start = time.time()

# ---------------- Spectrogram ----------------
print("\n" + "-"*50)
print("🔊 Spectrogram")
print("-"*50)

spec = get_spectrogram(src_name)

print(f"Min Value : {np.min(spec)}")
print(f"Max Value : {np.max(spec)}")
print(f"Shape     : {spec.shape}")
print(f"Preview   : \n{spec[:80]}")   # only first 2 rows

# ---------------- Peaks ----------------
print("\n" + "-"*50)
print("📈 Peak Detection")
print("-"*50)

peaks = find_peaks(spec)
save_json(f"{file_name1}.json", peaks, "peaks")

print(f"Total Peaks : {len(peaks)}")
print(f"Sample      : {peaks[:50]}")

# ---------------- Hashes ----------------
print("\n" + "-"*50)
print("🔐 Hash Generation")
print("-"*50)

hashes = generate_hashes(peaks)
save_json(f"{file_name1}.json", hashes, "hashes")

print(f"Total Hashes: {len(hashes)}")
print(f"Sample      : {hashes[:10]}")


# ---------------- Timing ----------------
time_end = time.time()

print("\n" + "-"*50)
print("⏱️ Performance")
print("-"*50)

print(f"Total Time  : {time_end - time_start:.3f} sec")

# ---------------- Plot ----------------
print("\n" + "-"*50)
print("📊 Visualization")
print("-"*50)

plot(spec)
print("Status      : Spectrogram plotted")

print("\n" + "="*50)
print("✅ DONE")
print("="*50 + "\n")
