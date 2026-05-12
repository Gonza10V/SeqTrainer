from pathlib import Path

from seqtrainer.genome_ntp.data import GenomeWindowDataset, interval_split, parse_fasta
from seqtrainer.genome_ntp.tokenizer import DNATokenizer


def test_tokenizer_normalize_and_encode():
    t = DNATokenizer()
    assert t.normalize("acgtx") == "ACGTN"
    assert t.encode("ACGTN") == [1, 2, 3, 4, 5]


def test_parse_fasta(tmp_path: Path):
    fp = tmp_path / "x.fna"
    fp.write_text(">chr1\nACGT\nNN\n")
    rec = parse_fasta(fp)
    assert rec["chr1"] == "ACGTNN"


def test_interval_split_no_overlap():
    sp = interval_split(10000, buffer_bp=100)
    assert sp.train[1] < sp.val[0]
    assert sp.val[1] < sp.test[0]


def test_windowing_shapes():
    toks = [1, 2, 3, 4, 5] * 100
    ds = GenomeWindowDataset(toks, (0, 200), chunk_len=32, stride=16)
    x, y = ds[0]
    assert len(x) == 32
    assert len(y) == 32
    assert x[1].item() == y[0].item()
