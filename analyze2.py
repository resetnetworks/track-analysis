import essentia.standard as es
import subprocess
import json
import time

from random.cacheInfo import get_cached_data, set_cache_data

class AudioAnalyzer:
    """Class to handle audio feature extraction using Essentia."""

    def __init__(self, filename):
        self.filename = filename
        # self.audio = es.MonoLoader(filename=filename)()
        # Load audio
        audio = es.MonoLoader(filename=filename)()
        # Ensure even length and float32
        self.audio = audio[:len(audio) - len(audio) % 2].astype('float32')
        self.data = {}

    def analyze_rhythm_and_key(self):
        tempo, beats, beats_confidence, _, _ = es.RhythmExtractor2013()(self.audio)
        key, scale, strength = es.KeyExtractor()(self.audio)
        self.data.update({
            "tempo": round(float(tempo), 2),
            "beats_count": int(len(beats)),
            "key": key,
            "scale": scale,
            "strength": strength
        })

    def analyze_energy_and_loudness(self):
        loudness = es.Loudness()(self.audio)
        energy = es.Energy()(self.audio)
        duration = es.Duration()(self.audio)
        rms = es.RMS()(self.audio)

        energy_pct = min(100, rms * 100)
        loudness_pct = min(100, (loudness / (duration * 10)) * 100)

        self.data.update({
            "duration": round(duration, 2),
            "loudness": round(float(loudness), 2),
            "loudness_percent": round(float(loudness_pct), 2),
            "energy": round(float(energy), 2),
            "energy_percent": round(float(energy_pct), 2)
        })

        

    def analyze_timbre_and_spectral(self):
        centroid = es.Centroid()(self.audio)
        zcr = es.ZeroCrossingRate()(self.audio)

        spectrum = es.Spectrum()
        spectral_centroid = es.SpectralCentroidTime()
        spec = spectrum(self.audio)
        brightness = spectral_centroid(spec)

        self.data.update({
            "brightness": round(float(brightness), 2),
            "centroid": round(float(centroid), 2),
            "noisiness": round(float(zcr), 2)
        })

    def run_all(self):
        self.data["file"] = self.filename
        self.analyze_rhythm_and_key()
        self.analyze_energy_and_loudness()
        self.analyze_timbre_and_spectral()
        return self.data


class ChromaPrint:
    """Class to handle chromaprint extraction (FPcalc)."""

    def __init__(self, filename):
        self.filename = filename
        self.fingerprint = None
        self.duration_fp = None

    def extract(self):
        try:
            result = subprocess.run(
                ["fpcalc", self.filename],
                capture_output=True,
                text=True
            )
            for line in result.stdout.splitlines():
                if line.startswith("FINGERPRINT="):
                    self.fingerprint = line.split("=")[1]
                if line.startswith("DURATION="):
                    self.duration_fp = int(line.split("=")[1])
        except FileNotFoundError:
            print("fpcalc not installed or not found in PATH.")
        return self.fingerprint, self.duration_fp


class AudioAnalysisManager:
    """Manages caching and analysis in an OOP structure."""

    def __init__(self, filename):
        self.filename = filename

    def get_analysis(self):
        """Return cached analysis if available, else run analysis and cache it."""
        cached = get_cached_data(self.filename)
        if cached:
            print("Using cached result.")
            return cached

        print("Cache not found. Running analysis...")
        analyzer = AudioAnalyzer(self.filename)
        data = analyzer.run_all()

        # Optional: Chromaprint can be included here
        chroma = ChromaPrint(self.filename)
        fingerprint, duration_fp = chroma.extract()
        if fingerprint:
            data["fingerprint"] = fingerprint
        if duration_fp:
            data["duration"] = duration_fp

        # Save to cache
        set_cache_data(self.filename, data)
        return data


if __name__ == "__main__":
    start_time = time.time()
    print(f"\n\n.........Analysis Started.........\n")

    filename = "songs/GIADAR - icarus by the sea - 04 touching you.wav"
    manager = AudioAnalysisManager(filename)
    result = manager.get_analysis()

    # Remove fingerprint from data for printing
    result.pop("fingerprint", None)

    print(json.dumps(result, indent=2))

    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    print(f"\n.........Analysis Ended.........\n\n")