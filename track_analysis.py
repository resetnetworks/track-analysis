"""
track_analysis.py

Core modular analyzer for processing individual audio files.
Extracts comprehensive acoustic features (rhythm, tonal, timbre, spectral) via Essentia,
computes embeddings, and runs inference on pre-trained models (EffNet, YAMNet) for 
classification (genre, danceability, instruments, vocal type, audio events).
Also extracts Chromaprint fingerprints.
"""

import essentia
import essentia.standard as es
import subprocess
import json
import numpy as np
import os
import time

def extract_rhythm_and_tonal(audio):
    tempo, beats, beats_confidence, _, _ = es.RhythmExtractor2013()(audio)
    key, scale, strength = es.KeyExtractor()(audio)
    
    return {
        "tempo": float(tempo),
        "beats_count": int(len(beats)),
        "beat_positions": beats.tolist() if isinstance(beats, np.ndarray) else list(beats),
        "beat_confidence": float(beats_confidence),
        "key": key,
        "scale": scale
    }

def extract_loudness_and_timbre(audio):
    loudness = es.Loudness()(audio)
    energy = es.Energy()(audio)
    duration = es.Duration()(audio)
    
    rms = es.RMS()(audio)
    energy_pct = min(100, rms * 100)
    loudness_pct = min(100, (loudness / (duration * 10)) * 100)
    
    centroid = es.Centroid()(audio)
    zcr = es.ZeroCrossingRate()(audio)
    
    spectrum = es.Spectrum()
    spectral_centroid = es.SpectralCentroidTime()
    spec = spectrum(audio)
    brightness = spectral_centroid(spec)
    
    return {
        "duration": float(duration),
        "loudness": float(loudness),
        "loudness_percent": float(loudness_pct),
        "energy": float(energy),
        "energy_percent": float(energy_pct),
        "rms": float(rms),
        "brightness": float(brightness),
        "centroid": float(centroid),
        "noisiness": float(zcr)
    }

def extract_advanced_features(audio):
    frame_size = 2048
    hop_size = 1024
    
    w = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    mfcc = es.MFCC()
    spectral_peaks = es.SpectralPeaks()
    hpcp = es.HPCP()
    spectral_contrast = es.SpectralContrast(frameSize=frame_size, sampleRate=44100)
    rolloff = es.RollOff()
    flatness = es.FlatnessDB()
    flux = es.Flux()
    dynamic_complexity, _ = es.DynamicComplexity()(audio)
    
    mfccs_list = []
    hpcps_list = []
    contrasts_list = []
    rolloffs_list = []
    flatness_list = []
    fluxes_list = []
    
    for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        spec = spectrum(w(frame))
        
        _, mfcc_coeffs = mfcc(spec)
        mfccs_list.append(mfcc_coeffs)
        
        freqs, mags = spectral_peaks(spec)
        hpcp_val = hpcp(freqs, mags)
        hpcps_list.append(hpcp_val)
        
        valleys, peaks = spectral_contrast(spec)
        contrasts_list.append(np.mean(peaks) - np.mean(valleys))
        
        rolloffs_list.append(rolloff(spec))
        flatness_list.append(flatness(spec))
        fluxes_list.append(flux(spec))
        
    return {
        "spectral_rolloff": float(np.mean(rolloffs_list)) if rolloffs_list else 0.0,
        "spectral_flux": float(np.mean(fluxes_list)) if fluxes_list else 0.0,
        "spectral_contrast": float(np.mean(contrasts_list)) if contrasts_list else 0.0,
        "spectral_flatness": float(np.mean(flatness_list)) if flatness_list else 0.0,
        "mfcc_mean": np.mean(mfccs_list, axis=0).tolist() if mfccs_list else [],
        "hpcp_mean": np.mean(hpcps_list, axis=0).tolist() if hpcps_list else [],
        "dynamic_complexity": float(dynamic_complexity)
    }

def extract_effnet_embeddings(filename, audio_16k):
    embedding_model = "models/discogs_track_embeddings-effnet-bs64-1.pb"
    embedding_extractor = es.TensorflowPredictEffnetDiscogs(
        graphFilename=embedding_model,
        output="PartitionedCall:1"
    )
    embeddings = embedding_extractor(audio_16k)
    
    # Save the embeddings
    os.makedirs("effNet_embeddings", exist_ok=True)
    emb_base_name = os.path.splitext(os.path.basename(filename))[0]
    emb_out_path = os.path.join("effNet_embeddings", f"{emb_base_name}_effNet_embedding.npy")
    np.save(emb_out_path, embeddings)
    
    return {"effnet_embedding_path": emb_out_path}, embeddings

def extract_abstract_classifiers(audio_16k, embeddings):
    # ---------- DANCEABILITY ----------
    dance_model = "models/danceability-discogs-effnet-1.pb"
    dance_predictor = es.TensorflowPredict2D(
        graphFilename=dance_model,
        input="model/Placeholder",
        output="model/Softmax"
    )
    dance_probs = dance_predictor(embeddings)
    dance_probs_mean = np.mean(dance_probs, axis=0)
    danceability = round(float(dance_probs_mean[0] * 100), 2)

    # ---------- GENRE ----------
    genre_model = "models/genre_discogs400-discogs-effnet-1.pb"
    genre_predictor = es.TensorflowPredict2D(
        graphFilename=genre_model,
        input="serving_default_model_Placeholder",
        output="PartitionedCall"
    )
    genre_probs = genre_predictor(embeddings)
    with open("models/genre_discogs400-discogs-effnet-1.json") as f:
        genre_meta = json.load(f)
    genre_labels = genre_meta["classes"]
    
    genre_probs_mean = np.mean(genre_probs, axis=0)
    genre_index = int(np.argmax(genre_probs_mean))
    genre_confidence = float(genre_probs_mean[genre_index] * 100)
    
    top3_idx = np.argsort(genre_probs_mean)[-3:][::-1]
    top_genres = [{"genre": genre_labels[i], "confidence": float(genre_probs_mean[i] * 100)} for i in top3_idx]

    # ---------- INSTRUMENT ----------
    instrument_model = "models/mtg_jamendo_instrument-discogs-effnet-1.pb"
    instrument_predictor = es.TensorflowPredict2D(
        graphFilename=instrument_model,
        input="model/Placeholder",
        output="model/Sigmoid"
    )
    instrument_probs = instrument_predictor(embeddings)
    with open("models/mtg_jamendo_instrument-discogs-effnet-1.json") as f:
        instrument_meta = json.load(f)
    instrument_labels = instrument_meta["classes"]
    
    instrument_probs_mean = np.mean(instrument_probs, axis=0)
    top_inst_idx = np.argsort(instrument_probs_mean)[-5:][::-1]
    top_instruments = [{"instrument": instrument_labels[i], "confidence": float(instrument_probs_mean[i] * 100)} for i in top_inst_idx]

    # ---------- VOICE / INSTRUMENTAL ----------
    voice_model = "models/voice_instrumental-discogs-effnet-1.pb"
    voice_predictor = es.TensorflowPredict2D(
        graphFilename=voice_model,
        input="model/Placeholder",
        output="model/Softmax"
    )
    voice_probs = voice_predictor(embeddings)
    voice_probs_mean = np.mean(voice_probs, axis=0)
    instrumental_confidence = float(voice_probs_mean[0] * 100)
    voice_confidence = float(voice_probs_mean[1] * 100)
    vocal_type = "voice" if voice_confidence > instrumental_confidence else "instrumental"

    # ---------- AUDISET YAMNET ----------
    yamnet_model = "models/audioset-yamnet-1.pb"
    yamnet_json = "models/audioset-yamnet-1.json"
    yamnet_extractor = es.TensorflowPredictVGGish(
        graphFilename=yamnet_model,
        input="melspectrogram",
        output="activations"
    )
    yamnet_probs = yamnet_extractor(audio_16k)
    with open(yamnet_json) as f:
        yamnet_meta = json.load(f)
    yamnet_labels = yamnet_meta["classes"]
    yamnet_probs_mean = np.mean(yamnet_probs, axis=0)
    
    top_event_indices = np.argsort(yamnet_probs_mean)[-5:][::-1]
    top_events = [{"event": yamnet_labels[i], "probability": float(yamnet_probs_mean[i])} for i in top_event_indices]
        
    speech_idx = yamnet_labels.index("Speech")
    music_idx = yamnet_labels.index("Music")
    singing_idx = yamnet_labels.index("Singing")
    
    avg_speech = float(np.mean(yamnet_probs[:, speech_idx])) * 100
    avg_music = float(np.mean(yamnet_probs[:, music_idx])) * 100
    avg_singing = float(np.mean(yamnet_probs[:, singing_idx])) * 100
    
    return {
        "danceability": danceability,
        "genre": genre_labels[genre_index],
        "genre_confidence": genre_confidence,
        "top_genres": top_genres,
        "instruments": top_instruments,
        "vocal_type": vocal_type,
        "voice_confidence": voice_confidence,
        "instrumental_confidence": instrumental_confidence,
        "vocal_type_yamnet": "voice" if (avg_singing + avg_speech) > 10 else "instrumental",
        "voice_confidence_yamnet": round(avg_singing + avg_speech, 2),
        "instrumental_confidence_yamnet": round(avg_music, 2),
        "audio_events": top_events,
        "Speech Probability": round(avg_speech, 2),
        "Music Probability": round(avg_music, 2),
        "Singing Probability": round(avg_singing, 2),
    }

def extract_chromaprint(filename):
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
            
    return {
        "fingerprint": "hidden on purpose",
        "chromaprint_duration": duration_fp
    }

def analyze_file(filename):
    print(f"\n======== Analyzing: {filename} ========")
    start_time = time.time()
    
    # Load audio
    audio = es.MonoLoader(filename=filename)()
    audio = audio[:len(audio) - len(audio) % 2].astype('float32')
    
    audio_16k = es.MonoLoader(filename=filename, sampleRate=16000)()
    audio_16k = audio_16k[:len(audio_16k) - len(audio_16k) % 2].astype('float32')

    # Extract Features
    rhythm_data = extract_rhythm_and_tonal(audio)
    loudness_data = extract_loudness_and_timbre(audio)
    advanced_data = extract_advanced_features(audio)
    effnet_data, embeddings = extract_effnet_embeddings(filename, audio_16k)
    fingerprint_data = extract_chromaprint(filename)
    
    # Optional abstract classifiers (you can disable this by returning {} or commenting it out)
    abstract_data = extract_abstract_classifiers(audio_16k, embeddings)
    # abstract_data = {}  # <--- Uncomment this to disable Group 2 classifiers
    
    # Update duration from Chromaprint if available, else keep Essentia's
    duration = fingerprint_data.get("chromaprint_duration") or loudness_data["duration"]
    loudness_data["duration"] = duration

    # Merge all dictionaries
    data = {
        "file": filename,
        **rhythm_data,
        **loudness_data,
        **advanced_data,
        **abstract_data,
        **effnet_data,
        "fingerprint": fingerprint_data["fingerprint"]
    }
    
    # Print the JSON output
    print(json.dumps(data, indent=2))
    
    # Save to disk
    os.makedirs("results", exist_ok=True)
    base_name = os.path.basename(filename)
    out_name = os.path.splitext(base_name)[0] + ".json"
    out_path = os.path.join("results", out_name)
    
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"Results successfully saved to {out_path}")
    
    end_time = time.time()
    print(f"Total runtime for {filename}: {end_time - start_time:.2f} seconds")
    print(f"\n........Analysis Ended.........\n\n")

if __name__ == "__main__":
    print(f"\n........Batch Analysis Started.........\n")
    overall_start_time = time.time()

    filenames = [
        "audio/GIADAR_album1/GIADAR - icarus by the sea - 01 the flight.wav",
        "new-audio/Yiruma, (이루마) - River Flows in You.mp3",
        "new-audio/Nirvana - Smells Like Teen Spirit (Official Music Video).mp3",
        "new-audio/Adele - Someone Like You (Official Music Video).mp3"
    ]

    for f in filenames:
        try:
            analyze_file(f)
        except Exception as e:
            print(f"Error analyzing {f}: {e}")

    overall_end_time = time.time()
    print(f"Total Batch Runtime: {overall_end_time - overall_start_time:.2f} seconds")