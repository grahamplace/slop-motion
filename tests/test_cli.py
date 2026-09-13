import json
import subprocess
import sys

from stop_motion.cli import main


def test_cli_json_contract(tmp_path, generator, monkeypatch, capsys):
    root = tmp_path / "cli project"
    brief = tmp_path / "brief.txt"
    brief.write_text("A clay butterfly.")
    assert main(["init", str(root), "--brief-file", str(brief)]) == 0
    initial = capsys.readouterr()
    assert json.loads(initial.out)["request_count"] == 0
    assert initial.err == ""
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Opening frame")
    monkeypatch.setattr("stop_motion.providers.openai.OpenAIResponses", lambda: generator)
    assert main(["frame", "--project", str(root), "--prompt-file", str(prompt)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["id"] == "f0001"
    assert output["warnings"] == []
    assert main(["view", "--project", str(root), "--frame", "f0001"]) == 0
    assert json.loads(capsys.readouterr().out)["frames"][0]["path"] == output["path"]
    assert main(["view", "--project", str(root), "--frame", "missing"]) == 1
    error = capsys.readouterr()
    assert error.out == ""
    assert "Unknown frame" in json.loads(error.err)["error"]


def test_fresh_process_can_read_project_without_api_key(project, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stop_motion",
            "status",
            "--project",
            str(project.root),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stderr == ""
    assert json.loads(result.stdout)["brief"] == "A clay butterfly lifts off."
