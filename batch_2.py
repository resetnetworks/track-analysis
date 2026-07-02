"""
batch_2.py
Date: 2026-07-02

This script processes an entire folder of audio files (an album) using the modular 
extraction functions from track_analysis.py. It aggregates the features of each track 
and saves the combined results into a single JSON file.
"""

import os
import time
import json
import essentia.standard as es
from track_analysis import (
    extract_rhythm_and_tonal,
    extract_loudness_and_timbre,
    extract_advanced_features,
    extract_effnet_embeddings,
    extract_abstract_classifiers,
    extract_chromaprint
)

def analyze_track(filename):
    start_time = time.time()
    print(f"\n--- Processing: {filename} ---")
    
    # Load audio
    audio = es.MonoLoader(filename=filename)()
    audio = audio[:len(audio) - len(audio) % 2].astype('float32')
    
    audio_16k = es.MonoLoader(filename=filename, sampleRate=16000)()
    audio_16k = audio_16k[:len(audio_16k) - len(audio_16k) % 2].astype('float32')

    # Extract all features using the modular functions from track_analysis.py
    rhythm_data = extract_rhythm_and_tonal(audio)
    loudness_data = extract_loudness_and_timbre(audio)
    advanced_data = extract_advanced_features(audio)
    effnet_data, embeddings = extract_effnet_embeddings(filename, audio_16k)
    fingerprint_data = extract_chromaprint(filename)
    
    abstract_data = extract_abstract_classifiers(audio_16k, embeddings)
    
    # Duration adjustment based on chromaprint
    duration = fingerprint_data.get("chromaprint_duration") or loudness_data["duration"]
    loudness_data["duration"] = duration

    end_time = time.time()
    track_time = end_time - start_time

    # Merge into one output dictionary for the track
    data = {
        "file": os.path.basename(filename),
        "processing_time_sec": round(track_time, 2),
        **rhythm_data,
        **loudness_data,
        **advanced_data,
        **abstract_data,
        **effnet_data,
        "fingerprint": fingerprint_data["fingerprint"]
    }
    
    print(f"Track analysis done in {track_time:.2f} sec")
    return data, track_time

def main():
    # Set this to the folder you want to batch process
    folder_path = "audio/GIADAR"
    supported_formats = (".wav", ".mp3", ".flac", ".m4a")

    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    results = []
    total_tracks = 0

    print(f"\n====== BATCH PROCESS STARTED FOR: {folder_path} ======\n")
    total_start = time.time()

    for file in sorted(os.listdir(folder_path)):
        if not file.lower().endswith(supported_formats):
            continue

        full_path = os.path.join(folder_path, file)

        try:
            data, track_time = analyze_track(full_path)

            results.append({
                "file": file,
                "analysis": data,
                "processing_time_sec": round(track_time, 2)
            })

            total_tracks += 1
            print(f"✅ Done: {file} ({track_time:.2f}s)\n")

        except Exception as e:
            print(f"✖ Error processing {file}: {e}\n")

    if total_tracks == 0:
        print("No supported audio files found in the directory.")
        return

    total_time = time.time() - total_start

    # Determine album name from the folder name
    album_name = os.path.basename(folder_path.rstrip("/"))
    
    output = {
        "Album_name": album_name,
        "total_tracks": total_tracks,
        "total_processing_time_sec": round(total_time, 2),
        "avg_per_track_sec": round(total_time / total_tracks, 2),
        "tracks": results
    }

    # Save to the results directory
    os.makedirs("results", exist_ok=True)
    output_file = f"results/{album_name}_batch2.json"

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n====== BATCH SUMMARY ======")
    print(f"Total tracks: {total_tracks}")
    print(f"Total time: {total_time:.2f} sec")
    print(f"Avg per track: {(total_time / total_tracks):.2f} sec")
    print(f"Saved to: {output_file}\n")


if __name__ == "__main__":
    main()
