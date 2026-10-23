import os

import run_mac


def test_token_is_created_once_and_kept_in_env_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TOOL_WORKER_TOKEN", raising=False)
    (tmp_path / ".env").write_text("FOO=1")  # ohne abschliessenden Zeilenumbruch
    run_mac.ensure_worker_token()
    token = os.environ["TOOL_WORKER_TOKEN"]
    assert len(token) == 64
    text = (tmp_path / ".env").read_text()
    assert text.startswith("FOO=1\n") and f"TOOL_WORKER_TOKEN={token}" in text

    monkeypatch.delenv(
        "TOOL_WORKER_TOKEN"
    )  # naechster Start: dasselbe Token aus der .env
    run_mac.ensure_worker_token()
    assert (
        os.environ["TOOL_WORKER_TOKEN"] == token
        and (tmp_path / ".env").read_text().count("TOOL_WORKER_TOKEN=") == 1
    )


def test_an_existing_token_in_the_env_file_is_not_replaced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TOOL_WORKER_TOKEN", raising=False)
    (tmp_path / ".env").write_text('TOOL_WORKER_TOKEN="vorhanden"\n')
    run_mac.ensure_worker_token()
    assert (
        os.environ["TOOL_WORKER_TOKEN"] == "vorhanden"
        and (tmp_path / ".env").read_text().count("TOOL_WORKER_TOKEN=") == 1
    )
