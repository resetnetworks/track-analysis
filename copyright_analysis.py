# this code seems to be working fine, however need to look into working

"""
=============================================================================
COPYRIGHT & METADATA ANALYSIS

PURPOSE:
To check if an audio track is copyrighted/registered and fetch its official 
metadata (Title, Artist, Release info).

HOW IT WORKS:
1. Fingerprinting: Runs `fpcalc` (Chromaprint) to generate a unique acoustic hash.
2. AcoustID Match: Sends the hash to AcoustID's API to find matches in their database.
3. MusicBrainz Fetch: If matched, it uses the MBID (MusicBrainz ID) to fetch 
   the official track details from the MusicBrainz API.

OUTPUT:
Generates a JSON file in `results_copyright/` containing the match score, 
artist, title, and diagnostic metadata.
=============================================================================
"""

import os
import time
import json
import subprocess
import numpy as np
import essentia
import essentia.standard as es
import requests
from dotenv import load_dotenv
load_dotenv()

from constants import speech_labels, music_labels, singing_labels
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ACOUSTID_API_KEY = os.getenv("ACOUSTID_API_KEY")
# print("KEY:", ACOUSTID_API_KEY)

# ---------- CHROMAPRINT ----------
def get_fingerprint(filename):
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

    return fingerprint, duration_fp

# -------------- MUSIC BRAINZ ----------------
'''this says:
if we have matching acousticId song in the database or not
'''
def lookup_musicbrainz(recording_id):
    url = f"https://musicbrainz.org/ws/2/recording/{recording_id}"

    params = {
        "fmt": "json",
        "inc": "artists+releases+release-groups+tags+genres"
    }

    headers = {
        "User-Agent": "TrackAnalysisApp/1.0 (your@email.com)"
    }

    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()

    print("music brainz: ", response, "/n")
    return response.json()

# ------------------------- AcoustId -------------------------
''' This says:
Did we find a match?
How confident?
What song it might be?
'''
def lookup_acoustid(track_fingerprint, duration):
    url = "https://api.acoustid.org/v2/lookup"

    params = {
        "client": ACOUSTID_API_KEY,
        "meta": "recordings artists release releasegroups tracks sources", # can remove tracks and sources - just pure noise
        "duration": duration,
        "fingerprint": track_fingerprint,
        "format": "json"
    }

    response = requests.get(url, params=params)
    data = response.json()

    print("ACOUSTID RESPONSE:", json.dumps(data, indent=2))
    return data

# -------------- AcoustId Enriched data ---------------
def build_acoustic_enriched(best, results, duration_fp):
    recordings = best.get("recordings", [])

    # Safe duration extraction
    matched_duration = (
        recordings[0].get("duration")
        if recordings and recordings[0].get("duration")
        else None
    )

    duration_diff = (
        abs(duration_fp - matched_duration)
        if matched_duration else None
    )

    return {
        # 🔑 Raw best match
        "id": best.get("id"),
        "score": best.get("score"),

        # 🔑 ALL candidate matches
        "all_matches": [
            {
                "id": r.get("id"),
                "score": r.get("score")
            }
            for r in results
        ],

        # 🔑 Recording details
        "recordings": [
            {
                "recording_id": rec.get("id"),
                "title": rec.get("title"),
                "duration": rec.get("duration"),

                "artists": [
                    artist.get("name")
                    for artist in rec.get("artists", [])
                ],

                "releasegroups": [
                    {
                        "id": rg.get("id"),
                        "title": rg.get("title"),
                        "type": rg.get("type")
                    }
                    for rg in rec.get("releasegroups", [])
                ],

                "releases": [
                    {
                        "id": rel.get("id"),
                        "title": rel.get("title"),
                        "date": rel.get("date")
                    }
                    for rel in rec.get("releases", [])
                ]
            }
            for rec in recordings
        ],

        # 🔑 Diagnostics
        "match_count": len(results),
        "recording_count": len(recordings),

        # 🔑 Duration comparison
        "input_duration": duration_fp,
        "matched_duration": matched_duration,
        "duration_diff": duration_diff
    }

# ------------ IDENTIFICATION | COPYRIGHT -----------------
def analyse_track_copyright(filename):
    fingerprint, duration_fp = get_fingerprint(filename)

    if not fingerprint or not duration_fp:
        print("No fingerprint available")
        return None

    print(f"\nFingerprint: {fingerprint[:50]}...")
    print(f"duration: {duration_fp} \n ")

    acoustic_data = lookup_acoustid(fingerprint, duration_fp)
    all_metadata = extract_all_metadata(acoustic_data)

    if acoustic_data.get("status") != "ok":
        return "No Data in the MusicBrianz library."

    results = acoustic_data.get("results", [])

    if not results: return {"match": False}

     # ✅ Always pick best by score
    best = max(results, key=lambda x: x.get("score", 0))
    score = best.get("score", 0)

    # ✅ Optional: filter weak matches
    if score < 0.7:
        return {"match": False}
    
     # -------- Extract basic metadata from ACOUSTIC --------
    recordings = best.get("recordings", [])

    title = "Unknown Title"
    artist = "Unknown Artist"
    mbid = None

    if recordings:
        # rec = recordings[0]
        rec = max(recordings, key=lambda r: r.get("sources", 0))
        title = rec.get("title", "Unknown Title")
        artists = rec.get("artists", [])
        artist = artists[0].get("name") if artists else "Unknown Artist"
        mbid = rec.get("id")  # 🔥 KEY STEP
        print("---------mbid:", mbid, "--------")

    # -------- Fetch MusicBrainz --------
    musicbrainz_data = {}

    if mbid:
        try:
            musicbrainz_data = lookup_musicbrainz(mbid)
        except Exception as e:
            print("MusicBrainz fetch failed:", e)

    # -------- Clean acoustic data (store only useful parts) --------
    acoustic_clean = {
        "id": best.get("id"),
        "score": score,
        "recording_id": mbid
    }

    acoustic_enriched = build_acoustic_enriched(best, results, duration_fp)

    # -------- Final response --------
    return {
        "file_name": filename,
        "duration": duration_fp,

        "match": True,
        "score": score,
        "title": title,
        "artist": artist,

        "acousticData": acoustic_clean,
        "acousticEnrichedData": acoustic_enriched,

        # 🔥 NEW (IMPORTANT)
        "all_recordings_debug": all_metadata,

        "musicBrainz": musicbrainz_data
    }

    '''
    best = results[0]

    # --- START OF NEW ADDED SECTION ---
    recordings = best.get("recordings", [])

    if recordings:
        rec = recordings[0]
        title = rec.get("title", "unknown title")
        artists = rec.get("artists", [])
        artist = artists[0].get("name") if artists else "Unknown Artist"
    else:
        # Fallback: AcoustID Track IDs don't work with MusicBrainz.
        # You need the MBID from 'recordingids' metadata.
        print("No direct metadata, checking for MBIDs...")
        
        # Check if we have recording IDs available
        if "recordings" in best and best["recordings"]:
             mbid = best["recordings"][0].get("id")
             mb_data = lookup_musicbrainz(mbid)
             title = mb_data.get("title")
             artist = mb_data.get("artist-credit", [{}])[0].get("name")
        else:
             title = "Unknown Title"
             artist = "Unknown Artist"
    # --- END OF NEW ADDED SECTION ---
    
    return {
        "file_name": filename,
        "fingerprint": fingerprint[:50],
        "duration": duration_fp,
        "Acoustic records": acoustic_data,

        "match": True,
        "score": best.get("score"),
        "title": title,
        "artist": artist
    }
    '''


# all data extraction
def extract_all_metadata(acoustic_data):
    results = acoustic_data.get("results", [])

    all_data = []

    for result in results:
        score = result.get("score")
        result_id = result.get("id")

        recordings = result.get("recordings", [])

        for rec in recordings:
            rec_data = {
                "result_id": result_id,
                "score": score,

                "recording_id": rec.get("id"),
                "title": rec.get("title"),
                "duration": rec.get("duration"),
                "sources": rec.get("sources"),

                "artists": [
                    artist.get("name")
                    for artist in rec.get("artists", [])
                ],

                "releasegroups": [
                    {
                        "id": rg.get("id"),
                        "title": rg.get("title"),
                        "type": rg.get("type")
                    }
                    for rg in rec.get("releasegroups", [])
                ],

                "releases": [
                    {
                        "id": rel.get("id"),
                        "title": rel.get("title"),
                        "date": rel.get("date")
                    }
                    for rel in rec.get("releases", [])
                ]
            }

            all_data.append(rec_data)

    return all_data

def generate_json(file_path, result):

    output_folder = "results_copyright"
    os.makedirs(output_folder, exist_ok=True)

    track_name = os.path.splitext(os.path.basename(file_path))[0] + ".json"

    output_file = os.path.join(output_folder, track_name)

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)


def main():
    file_path = "audio/Maggie Lindemann - Pretty Girl (Cheat Codes x CADE Remix) [Official Video].mp3"

    print("\n====== COPYRIGHT CHECK ======\n")
    print(f"File: {file_path}")

    result = {}

    try:
        data = analyse_track_copyright(file_path)

        if not data:
            print("❌ Failed to process file")
            result = {
                "status": "error",
                "message": "Processing failed",
                "file_name": file_path
            }

        elif not data.get("match"):
            print("No match found (likely original)")
            result = {
                "status": "no_match",
                **data
            }
        else:
            print("✅ MATCH FOUND")
            print(f"Title : {data.get('title')}")
            print(f"Artist: {data.get('artist')}")
            print(f"Score : {round(data.get('score', 0), 3)}")

            result = {
                "status": "success",
                **data
            }


    except Exception as e:
        print(f"❌ Error: {e}")
        result = {
            "status": "error",
            "message": str(e),
            "file_name": file_path
        }

    # 🔥 ALWAYS write JSON
    generate_json(file_path, result)

    print("\n====== DONE ======\n")

if __name__ == "__main__":
    main()
