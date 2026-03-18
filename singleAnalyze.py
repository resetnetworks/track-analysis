'''THIS IS NEW SECTION OF CODE'''
import essentia
import essentia.standard as es
import subprocess
import json
import numpy as np

import time
start_time = time.time()
print(f"\n........Analysis Started.........\n")

filename = "audio/GIADAR_album1/GIADAR - icarus by the sea - 01 the flight.wav"

# ---------- ESSENTIA ----------
audio = es.MonoLoader(filename=filename)()
# Convert to even length and float32 to avoid warnings
audio = audio[:len(audio) - len(audio) % 2].astype('float32')

tempo, beats, beats_confidence, _, _ = es.RhythmExtractor2013()(audio)
key, scale, strength = es.KeyExtractor()(audio)

# ---------- ENERGY / LOUDNESS ----------
loudness = es.Loudness()(audio)
energy = es.Energy()(audio)
duration = es.Duration()(audio)

rms = es.RMS()(audio)
energy_pct = min(100, rms * 100)
loudness_pct = min(100, (loudness / (duration * 10)) * 100)

# ---------- TIMBRE FEATURES ----------
centroid = es.Centroid()(audio)           # brightness
zcr = es.ZeroCrossingRate()(audio)        # noisiness

# ---------- SPECTRAL FEATURES ----------
spectrum = es.Spectrum()
spectral_centroid = es.SpectralCentroidTime()

spec = spectrum(audio)
brightness = spectral_centroid(spec)


# ---------- EMBEDDINGS ----------
embedding_model = "models/discogs_track_embeddings-effnet-bs64-1.pb"

embedding_extractor = es.TensorflowPredictEffnetDiscogs(
    graphFilename=embedding_model,
    output="PartitionedCall:1"
)

embeddings = embedding_extractor(audio)

# ---------- DANCEABILITY ----------
dance_model = "models/danceability-discogs-effnet-1.pb"

dance_predictor = es.TensorflowPredict2D(
    graphFilename=dance_model,
    input="model/Placeholder",
    output="model/Softmax"
)

danceability = round(float(dance_predictor(embeddings)[0][0] * 100), 2)

# ---------- GENRE ----------
genre_model = "models/genre_discogs400-discogs-effnet-1.pb"

genre_predictor = es.TensorflowPredict2D(
    graphFilename=genre_model,
    input="serving_default_model_Placeholder",
    output="PartitionedCall"
)

genre_probs = genre_predictor(embeddings)

# Load Genre Labels
with open("models/genre_discogs400-discogs-effnet-1.json") as f:
    genre_meta = json.load(f)

genre_labels = genre_meta["classes"]

print("genre_probs shape:", genre_probs.shape)
print("num labels:", len(genre_labels))

# Get Top Genre / Average predictions across frames
genre_probs_mean = np.mean(genre_probs, axis=0)

genre_index = int(np.argmax(genre_probs_mean))
genre = genre_labels[genre_index]
genre_confidence = float(genre_probs_mean[genre_index] * 100)

top3_idx = np.argsort(genre_probs_mean)[-3:][::-1]

top_genres = []
for i in top3_idx:
    top_genres.append({
        "genre": genre_labels[i],
        "confidence": float(genre_probs_mean[i] * 100)
    })

# ---------- INSTRUMENT ----------
instrument_model = "models/mtg_jamendo_instrument-discogs-effnet-1.pb"

instrument_predictor = es.TensorflowPredict2D(
    graphFilename=instrument_model,
    input="model/Placeholder",
    output="model/Sigmoid"
)

instrument_probs = instrument_predictor(embeddings)

# Load labels
with open("models/mtg_jamendo_instrument-discogs-effnet-1.json") as f:
    instrument_meta = json.load(f)

instrument_labels = instrument_meta["classes"]

# Average across frames
instrument_probs_mean = np.mean(instrument_probs, axis=0)

# Top 5 instruments
top_inst_idx = np.argsort(instrument_probs_mean)[-5:][::-1]

top_instruments = []
for i in top_inst_idx:
    top_instruments.append({
        "instrument": instrument_labels[i],
        "confidence": float(instrument_probs_mean[i] * 100)
    })

# ---------- VOICE / INSTRUMENTAL ----------
voice_model = "models/voice_instrumental-discogs-effnet-1.pb"

voice_predictor = es.TensorflowPredict2D(
    graphFilename=voice_model,
    input="model/Placeholder",
    output="model/Softmax"
)

voice_probs = voice_predictor(embeddings)

# Average across frames
voice_probs_mean = np.mean(voice_probs, axis=0)

voice_confidence = float(voice_probs_mean[0] * 100)
instrumental_confidence = float(voice_probs_mean[1] * 100)

# Determine label
vocal_type = "voice" if voice_confidence > instrumental_confidence else "instrumental"


# ----------------Voice Extraction
# ---------- AUDISET YAMNET (Event Recognition) ----------
yamnet_model = "models/audioset-yamnet-1.pb"
yamnet_json = "models/audioset-yamnet-1.json"

# YAMNet requires es.TensorflowPredictVGGish for its specific preprocessing
yamnet_extractor = es.TensorflowPredictVGGish(
    graphFilename=yamnet_model,
    input="melspectrogram",
    output="activations" # Use "activations" for probabilities, "embeddings" for features
)

# Run prediction
# Note: YAMNet expects 16kHz audio. es.MonoLoader usually defaults to 44.1k/48k. 
# For best results, ensure your 'audio' variable was loaded at 16000Hz.
yamnet_probs = yamnet_extractor(audio)

# Load YAMNet Labels
with open(yamnet_json) as f:
    yamnet_meta = json.load(f)
yamnet_labels = yamnet_meta["classes"]

# Average across frames to find the most prominent sounds in the file
yamnet_probs_mean = np.mean(yamnet_probs, axis=0)

# Get Top 5 detected audio events
top_event_indices = np.argsort(yamnet_probs_mean)[-5:][::-1]

top_events = []
for i in top_event_indices:
    top_events.append({
        "event": yamnet_labels[i],
        "probability": float(yamnet_probs_mean[i])
    })

# Add to your 'data' dictionary later:
# data["audio_events"] = top_events

# Find the indices for the categories you care about
speech_idx = yamnet_labels.index("Speech")
music_idx = yamnet_labels.index("Music")
singing_idx = yamnet_labels.index("Singing")

# Average the probabilities across the whole song
avg_speech = float(np.mean(yamnet_probs[:, speech_idx])) * 100
avg_music = float(np.mean(yamnet_probs[:, music_idx])) * 100
avg_singing = float(np.mean(yamnet_probs[:, singing_idx])) * 100


# ---------- CHROMAPRINT ----------
result = subprocess.run(
    ["fpcalc", filename],
    capture_output=True,
    text=True
)

fingerprint = None
duration_fp = None

for line in result.stdout.splitlines():
    if line.startswith("FINGERPRINT="):
        fingerprint = line.split("=")[1]
    if line.startswith("DURATION="):
        duration_fp = int(line.split("=")[1])

# ---------- FINAL JSON ----------
data = {
    "file": filename,
    "duration": duration_fp or duration,

    "tempo": float(tempo),
    "beats_count": int(len(beats)),

    "key": key,
    "scale": scale,

    "loudness": float(loudness),
    "loudness_percent": float(loudness_pct),

    "energy": float(energy),
    "energy_percent": float(energy_pct),

    "brightness": float(brightness),
    "centroid": float(centroid),
    "noisiness": float(zcr),

    "danceability": danceability,

    "genre": genre,
    "genre_confidence": genre_confidence,
    "top_genres": top_genres,

    "fingerprint": "hidden on purpose",

    "instruments": top_instruments,

    "vocal_type": vocal_type,
    "voice_confidence": voice_confidence,
    "instrumental_confidence": instrumental_confidence,

    "vocal_type_yamnet": vocal_type,
    "voice_confidence_yamnet": voice_confidence,
    "instrumental_confidence_yamnet": instrumental_confidence,

    "audio_events": top_events,

    "Speech Probability": round(avg_speech, 2),
    "Music Probability": round(avg_music, 2),
    "Singing Probability": round(avg_singing, 2),
}

print(json.dumps(data, indent=2))

end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")
print(f"\n........Analysis Ended.........\n\n")