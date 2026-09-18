from switchsafe import benchmark
from switchsafe.transcribe import Transcript


def test_labelled_benchmark_connects_entity_policy_and_accuracy(tmp_path, monkeypatch):
    class Fake:
        def __init__(self, model):
            pass

        def transcribe(self, path):
            return Transcript("block the card", 2, -.2, .01)

    monkeypatch.setattr(benchmark, "FasterWhisperTranscriber", Fake)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sample_id,audio_path,reference,speaker_id,critical_action,critical_entity_confidence,is_safe\na,a.wav,block my card,s1,true,0.4,false\n")
    result = benchmark.run(manifest, tmp_path / "output.json", "fake")
    assert result["corpus_wer"] == 1 / 3
    assert result["decision_counts"] == {"request_confirmation": 1}
    assert result["policy_metrics"]["unsafe_accepted"] == 0
