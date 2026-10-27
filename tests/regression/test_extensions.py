"""Extensions: packages reached as gatac.<name> and `gatac <command>` through entry points."""
import sys
from importlib.metadata import EntryPoint

import pytest

import gatac
import gatac.cli


@pytest.fixture
def ext(tmp_path, monkeypatch):
    """An installed extension "demo" (module gatac_demo) with one command."""
    (tmp_path / "gatac_demo.py").write_text(
        "def hello():\n    return 'hi'\n\n"
        "def register(subparsers):\n"
        "    p = subparsers.add_parser('demo')\n"
        "    p.add_argument('--n', type=int, default=1)\n"
        "    p.set_defaults(func=lambda args: print('demo ran', args.n))\n\n"
        "def broken(subparsers):\n    raise RuntimeError('boom')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    eps = {
        "gatac.ext": [EntryPoint("demo", "gatac_demo", "gatac.ext"),
                      EntryPoint("io", "gatac_demo", "gatac.ext")],
        "gatac.cli": [EntryPoint("demo", "gatac_demo:register", "gatac.cli"),
                      EntryPoint("bad", "gatac_demo:broken", "gatac.cli")],
    }
    fake = lambda group: eps.get(group, [])
    monkeypatch.setattr(gatac, "entry_points", fake)
    monkeypatch.setattr(gatac.cli, "entry_points", fake)
    yield
    monkeypatch.delitem(gatac.__dict__, "demo", raising=False)
    sys.modules.pop("gatac_demo", None)


def test_extension_namespace(ext):
    assert set(gatac.extensions()) == {"demo", "io"}
    assert "demo" in dir(gatac)
    assert gatac.demo.hello() == "hi"
    assert gatac.io.__name__ == "gatac.io"                       # core names win
    with pytest.raises(AttributeError):
        gatac.nothing_here


def test_extension_command(ext, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gatac", "demo", "--n", "3"])
    gatac.cli.main()
    out = capsys.readouterr()
    assert "demo ran 3" in out.out
    assert "extension 'bad'" in out.err and "boom" in out.err    # a broken extension only warns

