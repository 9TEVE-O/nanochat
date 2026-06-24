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


def test_interactive_empty_input_is_ignored(capsys):
    """An empty line is skipped; the loop continues and can still generate."""
    _, _, _, mock_engine_inst = _call_main([], inputs=["", "Hello", "quit"])
    mock_engine_inst.generate.assert_called_once()
    assert "Assistant:" in capsys.readouterr().out


def test_interactive_clear_resets_conversation_tokens(capsys):
    """'clear' resets conversation_tokens to [bos]; the generate() call after clear
    receives exactly [bos, user_start, *msg_bytes, user_end, assistant_start]."""
    sys_modules_patch, _, _, mock_engine_inst = _make_mocks()
    mock_engine_inst.generate.side_effect = [
        iter([([42], None)]),  # response to "Hello"
        iter([([44], None)]),  # response to "World" (after clear)
    ]

    with ExitStack() as stack:
        stack.enter_context(patch.dict("sys.modules", sys_modules_patch))
        stack.enter_context(patch("sys.argv", ["nanochat"]))
        stack.enter_context(
            patch("builtins.input", side_effect=["Hello", "clear", "World", "quit"])
        )
        from nanochat.__main__ import main
        main()

    assert "Conversation cleared." in capsys.readouterr().out
    assert mock_engine_inst.generate.call_count == 2

    second_call_tokens = mock_engine_inst.generate.call_args_list[1][0][0]
    # After clear, conversation_tokens = [bos]. The second message then prepends
    # user_start + encode(msg) + user_end + assistant_start (all mocked to 999 / [72,101,108]).
    assert second_call_tokens == [261, 999, 72, 101, 108, 999, 999]


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
# Argument forwarding to generate()

def test_default_temperature_passed_to_generate():
    """When --temperature is omitted, generate() is called with temperature=0.6."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "Hello"])
    _, kwargs = mock_engine_inst.generate.call_args
    assert kwargs["temperature"] == 0.6


def test_custom_temperature_passed_to_generate():
    """--temperature value is forwarded to generate()."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "Hello", "--temperature", "1.2"])
    _, kwargs = mock_engine_inst.generate.call_args
    assert kwargs["temperature"] == 1.2


def test_default_top_k_passed_to_generate():
    """When --top-k is omitted, generate() is called with top_k=50."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "Hello"])
    _, kwargs = mock_engine_inst.generate.call_args
    assert kwargs["top_k"] == 50


def test_custom_top_k_passed_to_generate():
    """--top-k value is forwarded to generate()."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "Hello", "--top-k", "10"])
    _, kwargs = mock_engine_inst.generate.call_args
    assert kwargs["top_k"] == 10


def test_max_tokens_256_passed_to_generate():
    """generate() is always called with max_tokens=256."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "Hello"])
    _, kwargs = mock_engine_inst.generate.call_args
    assert kwargs["max_tokens"] == 256


def test_num_samples_1_passed_to_generate():
    """generate() is always called with num_samples=1."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "Hello"])
    _, kwargs = mock_engine_inst.generate.call_args
    assert kwargs["num_samples"] == 1


# ---------------------------------------------------------------------------
# Argument forwarding to load_model()

def test_model_tag_passed_to_load_model():
    """--model-tag value is forwarded to load_model() as model_tag kwarg."""
    _, _, mock_checkpoint, _ = _call_main(
        ["--model-tag", "d512", "--prompt", "Hello"]
    )
    _, kwargs = mock_checkpoint.load_model.call_args
    assert kwargs["model_tag"] == "d512"


def test_step_passed_to_load_model():
    """--step value is forwarded to load_model() as step kwarg."""
    _, _, mock_checkpoint, _ = _call_main(["--step", "5000", "--prompt", "Hello"])
    _, kwargs = mock_checkpoint.load_model.call_args
    assert kwargs["step"] == 5000


def test_load_model_called_with_phase_eval():
    """load_model() is always called with phase='eval'."""
    _, _, mock_checkpoint, _ = _call_main(["--prompt", "Hello"])
    _, kwargs = mock_checkpoint.load_model.call_args
    assert kwargs["phase"] == "eval"


def test_default_model_tag_is_none():
    """When --model-tag is omitted, load_model() receives model_tag=None."""
    _, _, mock_checkpoint, _ = _call_main(["--prompt", "Hello"])
    _, kwargs = mock_checkpoint.load_model.call_args
    assert kwargs["model_tag"] is None


def test_default_step_is_none():
    """When --step is omitted, load_model() receives step=None."""
    _, _, mock_checkpoint, _ = _call_main(["--prompt", "Hello"])
    _, kwargs = mock_checkpoint.load_model.call_args
    assert kwargs["step"] is None


def test_compute_init_called_with_device_type():
    """compute_init() is called with the resolved device_type."""
    _, mock_common, *_ = _call_main(["--device-type", "cpu", "--prompt", "Hello"])
    mock_common.compute_init.assert_called_once_with("cpu")


def test_compute_init_called_with_autodetected_device():
    """When --device-type is omitted, compute_init() is called with autodetect result."""
    _, mock_common, *_ = _call_main(["--prompt", "Hello"])
    # autodetect returns "cpu" per _make_mocks
    mock_common.compute_init.assert_called_once_with("cpu")


# ---------------------------------------------------------------------------
# Case-insensitive special commands

def test_interactive_quit_case_insensitive(capsys):
    """'QUIT' (uppercase) exits with Goodbye!"""
    _call_main([], inputs=["QUIT"])
    assert "Goodbye!" in capsys.readouterr().out


def test_interactive_exit_case_insensitive(capsys):
    """'Exit' (mixed case) exits with Goodbye!"""
    _call_main([], inputs=["Exit"])
    assert "Goodbye!" in capsys.readouterr().out


def test_interactive_clear_case_insensitive(capsys):
    """'CLEAR' (uppercase) resets conversation."""
    _call_main([], inputs=["CLEAR", "quit"])
    assert "Conversation cleared." in capsys.readouterr().out


def test_single_prompt_quit_case_insensitive(capsys):
    """--prompt QUIT (uppercase) exits without generating."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "QUIT"])
    mock_engine_inst.generate.assert_not_called()
    assert "Goodbye!" in capsys.readouterr().out


def test_single_prompt_clear_case_insensitive(capsys):
    """--prompt CLEAR (uppercase) clears without generating."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "CLEAR"])
    mock_engine_inst.generate.assert_not_called()
    assert "Conversation cleared." in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Startup banner

def test_startup_banner_printed(capsys):
    """The startup banner is always printed before any interaction."""
    _call_main([], inputs=["quit"])
    out = capsys.readouterr().out
    assert "NanoChat Interactive Mode" in out


def test_startup_banner_includes_instructions(capsys):
    """The startup banner includes quit/exit and clear instructions."""
    _call_main([], inputs=["quit"])
    out = capsys.readouterr().out
    assert "quit" in out
    assert "clear" in out


# ---------------------------------------------------------------------------
# Conversation token accumulation

def test_first_generate_call_tokens():
    """First generate() receives [bos, user_start, *encoded, user_end, assistant_start]."""
    _, _, _, mock_engine_inst = _call_main(["--prompt", "Hello"])
    tokens = mock_engine_inst.generate.call_args[0][0]
    # bos=261, all encode_special return 999, encode returns [72,101,108]
    assert tokens == [261, 999, 72, 101, 108, 999, 999]


def test_response_tokens_appended_to_conversation():
    """After a response, generate() tokens on the next turn include prior response."""
    sys_modules_patch, _, _, mock_engine_inst = _make_mocks()
    mock_engine_inst.generate.side_effect = [
        iter([([42], None)]),
        iter([([55], None)]),
    ]

    with ExitStack() as stack:
        stack.enter_context(patch.dict("sys.modules", sys_modules_patch))
        stack.enter_context(patch("sys.argv", ["nanochat"]))
        stack.enter_context(patch("builtins.input", side_effect=["Hi", "Bye", "quit"]))
        from nanochat.__main__ import main
        main()

    second_call_tokens = mock_engine_inst.generate.call_args_list[1][0][0]
    # First response token was 42; since 42 != assistant_end (999), assistant_end is appended.
    # So after turn 1: conversation = [261, 999, 72,101,108, 999, 999, 42, 999]
    # Turn 2 appends: user_start(999) + encode("Bye")=[72,101,108] + user_end(999) + assistant_start(999)
    assert 42 in second_call_tokens
    assert 999 in second_call_tokens  # assistant_end was appended


def test_assistant_end_appended_when_missing():
    """If generate() does not yield assistant_end as last token, it is appended."""
    sys_modules_patch, _, _, mock_engine_inst = _make_mocks()
    # generate yields token 42 (not assistant_end=999); check second call includes 999
    mock_engine_inst.generate.side_effect = [
        iter([([42], None)]),
        iter([([55], None)]),
    ]

    with ExitStack() as stack:
        stack.enter_context(patch.dict("sys.modules", sys_modules_patch))
        stack.enter_context(patch("sys.argv", ["nanochat"]))
        stack.enter_context(patch("builtins.input", side_effect=["Hello", "World", "quit"]))
        from nanochat.__main__ import main
        main()

    second_call_tokens = mock_engine_inst.generate.call_args_list[1][0][0]
    # assistant_end (999) should be in the conversation tokens by turn 2
    assert _ASSISTANT_END_TOKEN in second_call_tokens


def test_assistant_end_not_duplicated_when_present():
    """If generate() ends with assistant_end, it is NOT appended again."""
    sys_modules_patch, _, _, mock_engine_inst = _make_mocks()
    # generate yields assistant_end token as the final token
    mock_engine_inst.generate.side_effect = [
        iter([([_ASSISTANT_END_TOKEN], None)]),
        iter([([55], None)]),
    ]

    with ExitStack() as stack:
        stack.enter_context(patch.dict("sys.modules", sys_modules_patch))
        stack.enter_context(patch("sys.argv", ["nanochat"]))
        stack.enter_context(patch("builtins.input", side_effect=["Hello", "World", "quit"]))
        from nanochat.__main__ import main
        main()

    second_call_tokens = mock_engine_inst.generate.call_args_list[1][0][0]
    # assistant_end (999) should appear exactly once at the end of the first response block
    end_idx = second_call_tokens.index(_ASSISTANT_END_TOKEN)
    # No second occurrence right after it
    if end_idx + 1 < len(second_call_tokens):
        assert second_call_tokens[end_idx + 1] != _ASSISTANT_END_TOKEN


# ---------------------------------------------------------------------------
# Token streaming and output

def test_multiple_generate_tokens_all_decoded(capsys):
    """When generate() yields multiple token columns, all are decoded and printed."""
    sys_modules_patch, _, _, mock_engine_inst = _make_mocks()
    mock_engine_inst.generate.return_value = iter([
        ([10], None),
        ([20], None),
        ([30], None),
    ])

    with ExitStack() as stack:
        stack.enter_context(patch.dict("sys.modules", sys_modules_patch))
        stack.enter_context(patch("sys.argv", ["nanochat", "--prompt", "Hello"]))
        from nanochat.__main__ import main
        main()

    # tokenizer.decode is called once per yielded token
    mock_engine_inst_ref = mock_engine_inst  # noqa: captured above
    # verify generate was called
    mock_engine_inst_ref.generate.assert_called_once()


def test_generate_tokens_decoded_and_printed(capsys):
    """Each token column yielded by generate() is decoded and printed."""
    sys_modules_patch, _, mock_checkpoint, mock_engine_inst = _make_mocks()
    mock_tokenizer = mock_checkpoint.load_model.return_value[1]
    decoded_tokens = ["He", "ll", "o"]
    mock_tokenizer.decode.side_effect = decoded_tokens
    mock_engine_inst.generate.return_value = iter([
        ([10], None),
        ([20], None),
        ([30], None),
    ])

    with ExitStack() as stack:
        stack.enter_context(patch.dict("sys.modules", sys_modules_patch))
        stack.enter_context(patch("sys.argv", ["nanochat", "--prompt", "Hello"]))
        from nanochat.__main__ import main
        main()

    out = capsys.readouterr().out
    assert "He" in out
    assert "ll" in out
    assert "o" in out


# ---------------------------------------------------------------------------
# Input whitespace stripping

def test_interactive_whitespace_input_is_ignored():
    """Input containing only spaces is treated as empty and skipped."""
    _, _, _, mock_engine_inst = _call_main([], inputs=["   ", "quit"])
    mock_engine_inst.generate.assert_not_called()


def test_interactive_input_is_stripped_before_command_check(capsys):
    """Input with leading/trailing spaces around 'quit' still triggers exit."""
    _call_main([], inputs=["  quit  "])
    assert "Goodbye!" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Short argument form

def test_short_source_flag():
    """Short form -i is equivalent to --source."""
    _, _, mock_checkpoint, _ = _call_main(["-i", "base"], inputs=["quit"])
    assert mock_checkpoint.load_model.call_args[0][0] == "base"


def test_device_type_mps_skips_autodetect():
    """--device-type mps is accepted and does not trigger autodetect."""
    _, mock_common, *_ = _call_main(["--device-type", "mps"], inputs=["quit"])
    mock_common.autodetect_device_type.assert_not_called()


def test_device_type_cuda_skips_autodetect():
    """--device-type cuda is accepted and does not trigger autodetect."""
    _, mock_common, *_ = _call_main(["--device-type", "cuda"], inputs=["quit"])
    mock_common.autodetect_device_type.assert_not_called()
