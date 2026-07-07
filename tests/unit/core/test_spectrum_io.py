import pytest

from bl03u_masstool.core.spectrum_io import read_spectrum


def test_read_spectrum_accepts_gb18030_text(tmp_path):
    path = tmp_path / "spectrum.txt"
    lines = ["能量:12.0 eV", "1 2", "2 4"]
    path.write_bytes("\n".join(lines).encode("gb18030"))

    spectrum = read_spectrum(path)

    assert spectrum.metadata_lines == ["能量:12.0 eV"]
    assert spectrum.x.tolist() == [1.0, 2.0]
    assert spectrum.y.tolist() == [2.0, 4.0]


def test_read_spectrum_rejects_undecodable_bytes(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"\xff\xfe\xfa\xfb\n1 2\n")

    with pytest.raises(UnicodeDecodeError, match="unable to decode spectrum file"):
        read_spectrum(path)
