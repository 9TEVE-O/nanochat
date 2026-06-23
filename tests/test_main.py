"""Tests for nanochat.__main__.main()

Run with: python -m pytest tests/test_main.py -v

All heavy deps (torch, nanochat.common, nanochat.engine, nanochat.checkpoint_manager)
are patched via sys.modules so no GPU or trained weights are needed.
"""
from contextlib import ExitStack
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers

_ASSISTANT_END_TOKEN = 999


def _make_mocks():
    """Return (sys_modules_patch, mock_common, mock_checkpoint, mock_engine_inst)."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_bos_token_id.return_value = 261
    mock_tokenizer.encode_special.return_value = _ASSISTANT_END_TOKEN
    mock_tokenizer.encode.return_value = [72, 101, 108]
    mock_tokenizer.decode.return_value = "Hi"

    mock_engine_inst = MagicMock()
    mock_engine_inst.generate.return_value = iter([([42], None)])

    mock_common = MagicMock()
    mock_common.compute_init.return_value = (None, None, None, None, "cpu")
    mock_common.autodetect_device_type.return_value = "cpu"

    mock_checkpoint = MagicMock()
    mock_checkpoint.load_model.return_value = (MagicMock(), mock_tokenizer, None)

    sys_modules_patch = {
        "torch": MagicMock(),
        "nanochat.common": mock_common,
        "nanochat.engine": MagicMock(Engine=MagicMock(return_value=mock_engine_inst)),
        "nanochat.checkpoint_manager": mock_checkpoint,
    }
    return sys_modules_patch, mock_common, mock_checkpoint, mock_engine_inst


def _call_main(argv, inputs=None):
    """Call main() with patched argv, sys.modules, and optionally builtins.input.

    argv   – CLI args without the program name
    inputs – None (no patch), a list of strings/exceptions, or a bare exception instance

    Returns (exit_code, mock_common, mock_checkpoint, mock_engine_inst).
    exit_code is None if main() returned normally.
    """
    sys_modules_patch, mock_common, mock_checkpoint, mock_engine_inst = _make_mocks()

    exit_code = None
    with ExitStack() as stack:
        stack.enter_context(patch.dict("sys.modules", sys_modules_patch))
        stack.enter_context(patch("sys.argv", ["nanochat"] + list(argv)))
        if inputs is not None:
            stack.enter_context(patch("builtins.input", side_effect=inputs))
        from nanochat.__main__ import main
        try:
            main()
        except SystemExit as exc:
            exit_code = exc.code

    return exit_code, mock_common, mock_checkpoint, mock_engine_inst


# ---------------------------------------------------------------------------
# Arg-parsing

def test_invalid_source_exits():
    """--source value outside choices causes argparse to exit with code 2."""
    exit_code, *_ = _call_main(["--source", "bad"])
    assert exit_code == 2


def test_invalid_device_type_exits():
    """--device-type value outside choices causes argparse to exit with code 2."""
    exit_code, *_ = _call_main(["--device-type", "tpu"])
    assert exit_code == 2


def test_default_source_is_sft():
    """When --source is omitted, load_model receives 'sft'."""
    _, _, mock_checkpoint, _ = _call_main([], inputs=["quit"])
    assert mock_checkpoint.load_model.call_args[0][0] == "sft"


def test_source_base_passed_to_load_model():
    _, _, mock_checkpoint, _ = _call_main(["--source", "base"], inputs=["quit"])
    assert mock_checkpoint.load_model.call_args[0][0] == "base"


def test_source_rl_passed_to_load_model():
    _, _, mock_checkpoint, _ = _call_main(["--source", "rl"], inputs=["quit"])
    assert mock_checkpoint.load_model.call_args[0][0] == "rl"


def test_explicit_device_skips_autodetect():
    """When --device-type is given, autodetect_device_type() must not be called."""
    _, mock_common, *_ = _call_main(["--device-type", "cpu"], inputs=["quit"])
    mock_common.autodetect_device_type.assert_not_called()


def test_no_device_flag_calls_autodetect():
    """When --device-type is omitted, autodetect_device_type() is called once."""
    _, mock_common, *_ = _call_main([], inputs=["quit"])
    mock_common.autodetect_device_type.assert_called_once()


# ---------------------------------------------------------------------------
# Single-prompt mode  (--prompt given)

def test_single_prompt_triggers_generation():
    """A non-empty --prompt runs generate() exactly once then exits."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "Hello"])
    mock_engine_inst.generate.assert_called_once()


def test_single_prompt_whitespace_skips_generation():
    """A whitespace-only prompt exits immediately without generating."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "   "])
    mock_engine_inst.generate.assert_not_called()


def test_single_prompt_quit_prints_goodbye(capsys):
    _call_main(["--prompt", "quit"])
    assert "Goodbye!" in capsys.readouterr().out


def test_single_prompt_quit_skips_generation():
    _, _, _, mock_engine_inst = _call_main(["--prompt", "quit"])
    mock_engine_inst.generate.assert_not_called()


def test_single_prompt_exit_skips_generation():
    _, _, _, mock_engine_inst = _call_main(["--prompt", "exit"])
    mock_engine_inst.generate.assert_not_called()


def test_single_prompt_clear_skips_generation(capsys):
    """--prompt clear resets conversation without generating, then exits."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "clear"])
    mock_engine_inst.generate.assert_not_called()
    assert "Conversation cleared." in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Interactive mode  (no --prompt, input() called)

def test_interactive_quit_exits(capsys):
    _call_main([], inputs=["quit"])
    assert "Goodbye!" in capsys.readouterr().out


def test_interactive_exit_exits(capsys):
    _call_main([], inputs=["exit"])
    assert "Goodbye!" in capsys.readouterr().out


def test_interactive_eoferror_exits(capsys):
    _call_main([], inputs=EOFError())
    assert "Goodbye!" in capsys.readouterr().out


def test_interactive_keyboard_interrupt_exits(capsys):
    _call_main([], inputs=KeyboardInterrupt())
    assert "Goodbye!" in capsys.readouterr().out


def test_interactive_empty_input_is_ignored():
    """An empty line is skipped; the loop continues until 'quit'."""
    _, _, _, mock_engine_inst = _call_main([], inputs=["", "quit"])
    mock_engine_inst.generate.assert_not_called()


def test_interactive_clear_resets_conversation(capsys):
    _, _, _, mock_engine_inst = _call_main([], inputs=["clear", "quit"])
    mock_engine_inst.generate.assert_not_called()
    assert "Conversation cleared." in capsys.readouterr().out


def test_interactive_message_generates_response(capsys):
    _, _, _, mock_engine_inst = _call_main([], inputs=["Hello", "quit"])
    mock_engine_inst.generate.assert_called_once()
    assert "Assistant:" in capsys.readouterr().out


def test_interactive_multi_turn():
    """Two user messages result in two separate generate() calls."""
    sys_modules_patch, _, _, mock_engine_inst = _make_mocks()
    call_count = 0

    def _fresh_iter(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return iter([([42], None)])

    mock_engine_inst.generate.side_effect = _fresh_iter

    with ExitStack() as stack:
        stack.enter_context(patch.dict("sys.modules", sys_modules_patch))
        stack.enter_context(patch("sys.argv", ["nanochat"]))
        stack.enter_context(patch("builtins.input", side_effect=["Hello", "World", "quit"]))
        from nanochat.__main__ import main
        main()

    assert call_count == 2
