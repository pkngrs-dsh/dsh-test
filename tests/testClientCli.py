"""Startup selection without model loading or opening a network listener."""

import json
import sys

import pytest

from faceAppearance import cli, service


@pytest.mark.parametrize("command,assembly", [("serve-client", True), ("serve", False)])
def testServiceCommandSelectsConfigAndForwardsSettings(monkeypatch, tmp_path, command, assembly):
    import uvicorn
    captured = {}
    app = object()

    def create(config, **kwargs):
        captured.update(config=config, **kwargs)
        return app

    def run(actual, **kwargs):
        assert actual is app
        captured["uvicorn"] = kwargs

    monkeypatch.setattr(service, "createApp", create)
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setenv("TEST_CLIENT_TOKEN", "fixture-secret")
    monkeypatch.setattr(sys, "argv", ["face-appearance", command, "--queueCapacity", "3",
                        "--outputRoot", str(tmp_path), "--apiTokenEnv", "TEST_CLIENT_TOKEN"])
    cli.main()
    assert captured["config"].enableAssembly is assembly
    assert captured["config"].workingResolution == (1024 if assembly else 512)
    assert captured["apiToken"] == "fixture-secret"
    assert captured["queueCapacity"] == 3
    assert captured["outputRoot"] == tmp_path
    assert captured["uvicorn"] == {"host": "127.0.0.1", "port": 8000, "workers": 1}


def testClientCommandRejectsConfigWithoutAssembly(monkeypatch, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"enableAssembly": False}))
    monkeypatch.setattr(sys, "argv", ["face-appearance", "--config", str(path), "serve-client"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
