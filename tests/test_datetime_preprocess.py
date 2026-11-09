import json

import pytest

from rt.preprocess import _preprocess as pre


@pytest.mark.parametrize("valid", [False, True])
def test_bulk_exact_reuse_requires_content_validation(tmp_path, monkeypatch, valid):
    db = tmp_path / "example"
    db.mkdir()
    (db / "meta.json").write_text(
        json.dumps({"text_embeddings": {"test": {"file": "text.bin"}}})
    )
    (db / "text.bin").touch()
    # Both filenames exist even for the invalid/partial publication case.
    (db / "datetime_exact.json").write_text("{}")
    (db / "datetime_exact.bin").touch()
    monkeypatch.setattr(
        pre, "list_datasets", lambda *args, **kwargs: ["org/collection/example"]
    )
    validated = []

    def validate(*, pre_dir):
        validated.append(pre_dir)
        if not valid:
            raise ValueError("invalid exact datetime payload length")

    monkeypatch.setattr(pre, "validate_datetime", validate)

    def run():
        pre.many(
            repo="org/collection",
            out_dir=str(tmp_path),
            shard=0,
            num_shards=1,
            skip_existing=True,
            embedder="test",
            batch_size=1,
            skip_tasks=False,
            embed=True,
            upload_repo=None,
            public=False,
            revision=None,
            exact_datetime=True,
        )

    if valid:
        run()
    else:
        with pytest.raises(SystemExit) as error:
            run()
        assert error.value.code == 1
    assert validated == [str(db)]


def test_fresh_exact_preprocessing_enters_native_provenance_boundary(
    tmp_path, monkeypatch
):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "manifest.yaml").write_text("name: example\n")
    calls = []
    monkeypatch.setattr(pre, "preprocess", lambda *args, **kw: calls.append((args, kw)))
    pre.preprocess_one(
        str(raw),
        tmp_path / "output",
        embedder="unused",
        batch_size=1,
        skip_tasks=False,
        embed=False,
        upload_repo=None,
        private=True,
        revision=None,
        exact_datetime=True,
    )
    assert len(calls) == 1
    assert calls[0][1]["exact_datetime"] is True
    # Native preprocessing captures source identities before archive construction;
    # a separate Python post-hoc backfill cannot provide that guarantee.
