import pytest

from switchsafe.dataset import read_manifest


def test_manifest_preserves_critical_signals(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("sample_id,audio_path,reference,speaker_id,critical_action,critical_entity_confidence,is_safe\na,a.wav,block account,s1,true,0.4,false\n")
    sample = read_manifest(path)[0]
    assert sample.critical_action
    assert sample.critical_entity_confidence == .4
    assert sample.is_safe is False


def test_blank_reference_rejected(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("sample_id,audio_path,reference\na,a.wav,\n")
    with pytest.raises(ValueError, match="independent"):
        read_manifest(path)
