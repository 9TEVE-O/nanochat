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
    # Distinct IDs per special token so tests can count assistant_end (999) unambiguously.
    _special = {
        "<|user_start|>": 901,
        "<|user_end|>": 902,
        "<|assistant_start|>": 903,
        "<|assistant_end|>": _ASSISTANT_END_TOKEN,  # 999
    }
    mock_tokenizer.encode_special.side_effect = lambda name: _special[name]
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


def test_model_tag_passed_to_load_model():
    """--model-tag is forwarded as model_tag kwarg to load_model."""
    _, _, mock_checkpoint, _ = _call_main(["--model-tag", "v1"], inputs=["quit"])
    assert mock_checkpoint.load_model.call_args[1].get("model_tag") == "v1"


def test_step_passed_to_load_model():
    """--step is forwarded as step kwarg to load_model."""
    _, _, mock_checkpoint, _ = _call_main(["--step", "500"], inputs=["quit"])
    assert mock_checkpoint.load_model.call_args[1].get("step") == 500


def test_default_model_tag_is_none():
    """When --model-tag is omitted, load_model receives model_tag=None."""
    _, _, mock_checkpoint, _ = _call_main([], inputs=["quit"])
    assert mock_checkpoint.load_model.call_args[1].get("model_tag") is None


def test_default_step_is_none():
    """When --step is omitted, load_model receives step=None."""
    _, _, mock_checkpoint, _ = _call_main([], inputs=["quit"])
    assert mock_checkpoint.load_model.call_args[1].get("step") is None


def test_load_model_phase_is_eval():
    """load_model is always called with phase='eval'."""
    _, _, mock_checkpoint, _ = _call_main([], inputs=["quit"])
    assert mock_checkpoint.load_model.call_args[1].get("phase") == "eval"


def test_compute_init_called_exactly_once():
    """compute_init() is called exactly once per main() invocation."""
    _, mock_common, *_ = _call_main([], inputs=["quit"])
    mock_common.compute_init.assert_called_once()


def test_load_model_called_exactly_once():
    """load_model() is called exactly once per main() invocation."""
    _, _, mock_checkpoint, _ = _call_main([], inputs=["quit"])
    mock_checkpoint.load_model.assert_called_once()


# ---------------------------------------------------------------------------
# Device-type handling

def test_device_type_cuda_is_valid():
    """--device-type cuda is a valid choice; argparse must not exit with code 2."""
    exit_code, *_ = _call_main(["--device-type", "cuda"], inputs=["quit"])
    assert exit_code != 2


def test_device_type_mps_is_valid():
    """--device-type mps is a valid choice; argparse must not exit with code 2."""
    exit_code, *_ = _call_main(["--device-type", "mps"], inputs=["quit"])
    assert exit_code != 2


def test_explicit_device_type_passed_to_compute_init():
    """The explicit --device-type value is passed to compute_init()."""
    _, mock_common, *_ = _call_main(["--device-type", "cpu"], inputs=["quit"])
    assert mock_common.compute_init.call_args[0][0] == "cpu"


def test_autodetected_device_type_passed_to_compute_init():
    """The autodetected device type is passed to compute_init()."""
    _, mock_common, *_ = _call_main([], inputs=["quit"])
    assert mock_common.compute_init.call_args[0][0] == "cpu"


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


def test_single_prompt_exit_prints_goodbye(capsys):
    """--prompt exit prints 'Goodbye!' just like --prompt quit."""
    _call_main(["--prompt", "exit"])
    assert "Goodbye!" in capsys.readouterr().out


def test_single_prompt_clear_skips_generation(capsys):
    """--prompt clear resets conversation without generating, then exits."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "clear"])
    mock_engine_inst.generate.assert_not_called()
    assert "Conversation cleared." in capsys.readouterr().out


def test_single_prompt_quit_uppercase(capsys):
    """--prompt QUIT (uppercase) prints Goodbye and skips generation."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "QUIT"])
    mock_engine_inst.generate.assert_not_called()
    assert "Goodbye!" in capsys.readouterr().out


def test_single_prompt_exit_uppercase_skips_generation():
    """--prompt EXIT (uppercase) skips generation."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "EXIT"])
    mock_engine_inst.generate.assert_not_called()


def test_single_prompt_clear_uppercase(capsys):
    """--prompt CLEAR (uppercase) clears conversation without generating."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "CLEAR"])
    mock_engine_inst.generate.assert_not_called()
    assert "Conversation cleared." in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Generate kwargs

def test_temperature_passed_to_generate():
    """--temperature value is forwarded to engine.generate()."""
    _, _, _, mock_engine_inst = _call_main(["--temperature", "0.9", "--prompt", "Hello"])
    assert mock_engine_inst.generate.call_args[1].get("temperature") == 0.9


def test_top_k_passed_to_generate():
    """--top-k value is forwarded to engine.generate()."""
    _, _, _, mock_engine_inst = _call_main(["--top-k", "100", "--prompt", "Hello"])
    assert mock_engine_inst.generate.call_args[1].get("top_k") == 100


def test_default_temperature_passed_to_generate():
    """Default temperature (0.6) is forwarded to engine.generate()."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "Hello"])
    assert mock_engine_inst.generate.call_args[1].get("temperature") == 0.6


def test_default_top_k_passed_to_generate():
    """Default top-k (50) is forwarded to engine.generate()."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "Hello"])
    assert mock_engine_inst.generate.call_args[1].get("top_k") == 50


def test_generate_called_with_num_samples_1():
    """engine.generate() is always called with num_samples=1."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "Hello"])
    assert mock_engine_inst.generate.call_args[1].get("num_samples") == 1


def test_generate_called_with_max_tokens_256():
    """engine.generate() is always called with max_tokens=256."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "Hello"])
    assert mock_engine_inst.generate.call_args[1].get("max_tokens") == 256


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


def test_interactive_quit_uppercase(capsys):
    """'QUIT' (uppercase) is treated the same as 'quit'."""
    _call_main([], inputs=["QUIT"])
    assert "Goodbye!" in capsys.readouterr().out


def test_interactive_exit_uppercase(capsys):
    """'EXIT' (uppercase) is treated the same as 'exit'."""
    _call_main([], inputs=["EXIT"])
    assert "Goodbye!" in capsys.readouterr().out


def test_interactive_clear_uppercase(capsys):
    """'CLEAR' (uppercase) resets the conversation."""
    _call_main([], inputs=["CLEAR", "quit"])
    assert "Conversation cleared." in capsys.readouterr().out


def test_interactive_empty_input_is_ignored(capsys):
    """An empty line is skipped; the loop continues and can still generate."""
    _, _, _, mock_engine_inst = _call_main([], inputs=["", "Hello", "quit"])
    mock_engine_inst.generate.assert_called_once()
    assert "Assistant:" in capsys.readouterr().out


def test_interactive_whitespace_only_input_is_ignored():
    """A whitespace-only interactive input is treated as empty and skipped."""
    _, _, _, mock_engine_inst = _call_main([], inputs=["   ", "Hello", "quit"])
    mock_engine_inst.generate.assert_called_once()


def test_interactive_multiple_empty_lines_all_skipped():
    """Multiple consecutive empty lines are all ignored before a real message."""
    _, _, _, mock_engine_inst = _call_main([], inputs=["", "", "", "Hello", "quit"])
    mock_engine_inst.generate.assert_called_once()


def test_interactive_clear_resets_conversation_tokens(capsys):
    """'clear' resets conversation_tokens to [bos]; the generate() call after clear
    receives exactly [bos, user_start, *msg_bytes, user_end, assistant_start]."""
    sys_modules_patch, _, _, mock_engine_inst = _make_mocks()

    # Capture a snapshot at call-time so post-call mutations don't affect the assertion.
    captured_tokens = []

    def _capturing_generate(tokens, **kwargs):
        captured_tokens.append(list(tokens))
        return iter([([42], None)])

    mock_engine_inst.generate.side_effect = _capturing_generate

    with ExitStack() as stack:
        stack.enter_context(patch.dict("sys.modules", sys_modules_patch))
        stack.enter_context(patch("sys.argv", ["nanochat"]))
        stack.enter_context(
            patch("builtins.input", side_effect=["Hello", "clear", "World", "quit"])
        )
        from nanochat.__main__ import main
        main()

    assert "Conversation cleared." in capsys.readouterr().out
    assert len(captured_tokens) == 2
    # After clear: bos=261, user_start=901, encode(World)=[72,101,108], user_end=902, assistant_start=903
    assert captured_tokens[1] == [261, 901, 72, 101, 108, 902, 903]


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


# ---------------------------------------------------------------------------
# Startup banner

def test_startup_banner_printed(capsys):
    """main() prints the NanoChat Interactive Mode banner on startup."""
    _call_main([], inputs=["quit"])
    assert "NanoChat Interactive Mode" in capsys.readouterr().out


def test_startup_banner_contains_instructions(capsys):
    """Banner mentions quit/exit and clear commands."""
    _call_main([], inputs=["quit"])
    out = capsys.readouterr().out
    assert "quit" in out
    assert "clear" in out


# ---------------------------------------------------------------------------
# Token-stream and conversation-state

def test_assistant_end_appended_when_missing():
    """When the last generated token is not assistant_end, it is appended.
    Verified by checking the second turn receives the appended token."""
    sys_modules_patch, _, _, mock_engine_inst = _make_mocks()
    captured: list[list[int]] = []

    def _gen(tokens, **kwargs):
        captured.append(list(tokens))
        return iter([([42], None)])

    mock_engine_inst.generate.side_effect = _gen

    with ExitStack() as stack:
        stack.enter_context(patch.dict("sys.modules", sys_modules_patch))
        stack.enter_context(patch("sys.argv", ["nanochat"]))
        stack.enter_context(patch("builtins.input", side_effect=["Hello", "World", "quit"]))
        from nanochat.__main__ import main
        main()

    assert _ASSISTANT_END_TOKEN in captured[1]
    assert 42 in captured[1]


def test_assistant_end_not_duplicated_when_present():
    """When the last generated token equals assistant_end (999), it is NOT appended again.
    Run 1 (response=assistant_end) contributes 1 token to conversation; Run 2 (response=42)
    contributes 2 tokens (42 + appended 999), so the second call's snapshot is 1 token shorter."""
    # Run 1: response IS assistant_end — must NOT be appended again (1 response token total)
    sp1, _, _, mei1 = _make_mocks()
    cap1: list[list[int]] = []

    def _gen1(tokens, **kwargs):
        cap1.append(list(tokens))
        return iter([([_ASSISTANT_END_TOKEN], None)]) if len(cap1) == 1 else iter([([42], None)])

    mei1.generate.side_effect = _gen1
    with ExitStack() as s:
        s.enter_context(patch.dict("sys.modules", sp1))
        s.enter_context(patch("sys.argv", ["nanochat"]))
        s.enter_context(patch("builtins.input", side_effect=["Hello", "World", "quit"]))
        from nanochat.__main__ import main
        main()

    # Run 2: response is NOT assistant_end — 999 IS appended (2 response tokens total)
    sp2, _, _, mei2 = _make_mocks()
    cap2: list[list[int]] = []

    def _gen2(tokens, **kwargs):
        cap2.append(list(tokens))
        return iter([([42], None)])

    mei2.generate.side_effect = _gen2
    with ExitStack() as s:
        s.enter_context(patch.dict("sys.modules", sp2))
        s.enter_context(patch("sys.argv", ["nanochat"]))
        s.enter_context(patch("builtins.input", side_effect=["Hello", "World", "quit"]))
        from nanochat.__main__ import main
        main()

    # Snapshots taken before post-call mutation; Run 2 second call is exactly 1 token longer.
    assert len(cap1[1]) == len(cap2[1]) - 1


def test_conversation_tokens_accumulate_across_turns():
    """Tokens from turn 1 response are present in the token list for turn 2."""
    sys_modules_patch, _, _, mock_engine_inst = _make_mocks()
    captured: list[list[int]] = []

    def _gen(tokens, **kwargs):
        captured.append(list(tokens))
        return iter([([77], None)]) if len(captured) == 1 else iter([([42], None)])

    mock_engine_inst.generate.side_effect = _gen

    with ExitStack() as stack:
        stack.enter_context(patch.dict("sys.modules", sys_modules_patch))
        stack.enter_context(patch("sys.argv", ["nanochat"]))
        stack.enter_context(patch("builtins.input", side_effect=["Hello", "World", "quit"]))
        from nanochat.__main__ import main
        main()

    assert len(captured) == 2
    assert 77 in captured[1]


def test_tokenizer_decode_called_per_streamed_token(capsys):
    """tokenizer.decode() is called once for each token yielded by generate(),
    and the decoded text is printed to stdout."""
    sys_modules_patch, _, _, mock_engine_inst = _make_mocks()
    mock_engine_inst.generate.return_value = iter([
        ([10], None),
        ([20], None),
        ([30], None),
    ])

    with ExitStack() as stack:
        stack.enter_context(patch.dict("sys.modules", sys_modules_patch))
        stack.enter_context(patch("sys.argv", ["nanochat"]))
        stack.enter_context(patch("builtins.input", side_effect=["Hello", "quit"]))
        from nanochat.__main__ import main
        main()

    mock_tokenizer = sys_modules_patch["nanochat.checkpoint_manager"].load_model.return_value[1]
    assert mock_tokenizer.decode.call_count == 3
    # decode.return_value is "Hi" (set in _make_mocks); verify it actually reaches stdout.
    assert capsys.readouterr().out.count("Hi") == 3


# ---------------------------------------------------------------------------
# Engine instantiation

def test_engine_instantiated_with_model_and_tokenizer():
    """Engine is constructed with the model and tokenizer returned by load_model."""
    sys_modules_patch, _, mock_checkpoint, _ = _make_mocks()
    mock_model = MagicMock()
    mock_tokenizer_inst = MagicMock()
    mock_tokenizer_inst.get_bos_token_id.return_value = 261
    mock_tokenizer_inst.encode_special.return_value = 999
    mock_tokenizer_inst.encode.return_value = [1, 2, 3]
    mock_tokenizer_inst.decode.return_value = "x"
    mock_checkpoint.load_model.return_value = (mock_model, mock_tokenizer_inst, None)

    mock_engine_cls = sys_modules_patch["nanochat.engine"].Engine

    with ExitStack() as stack:
        stack.enter_context(patch.dict("sys.modules", sys_modules_patch))
        stack.enter_context(patch("sys.argv", ["nanochat"]))
        stack.enter_context(patch("builtins.input", side_effect=["quit"]))
        from nanochat.__main__ import main
        main()

    mock_engine_cls.assert_called_once_with(mock_model, mock_tokenizer_inst)
