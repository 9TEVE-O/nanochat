"""CLI entry point — `nanochat` command after `pip install -e .`"""
import argparse


def main() -> None:
    import torch  # noqa: F401  (triggers CUDA init before other nanochat imports)
    from nanochat.common import compute_init, autodetect_device_type
    from nanochat.engine import Engine
    from nanochat.checkpoint_manager import load_model

    parser = argparse.ArgumentParser(description="Chat with a nanochat model")
    parser.add_argument("-i", "--source", type=str, default="sft",
                        help="Model source: sft|rl")
    parser.add_argument("-g", "--model-tag", type=str, default=None,
                        help="Model tag to load")
    parser.add_argument("-s", "--step", type=int, default=None,
                        help="Checkpoint step to load")
    parser.add_argument("-p", "--prompt", type=str, default="",
                        help="Single prompt (non-interactive)")
    parser.add_argument("-t", "--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("-k", "--top-k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--device-type", type=str, default="",
                        choices=["cuda", "cpu", "mps"],
                        help="Device: cuda|cpu|mps. Empty => autodetect")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, _ = load_model(
        args.source, device, phase="eval",
        model_tag=args.model_tag, step=args.step,
    )

    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    engine = Engine(model, tokenizer)

    print("\nNanoChat Interactive Mode")
    print("-" * 50)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    print("-" * 50)

    conversation_tokens = [bos]

    while True:
        if args.prompt:
            user_input = args.prompt
        else:
            try:
                user_input = input("\nUser: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            conversation_tokens = [bos]
            print("Conversation cleared.")
            continue

        if not user_input:
            continue

        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(user_input))
        conversation_tokens.append(user_end)
        conversation_tokens.append(assistant_start)

        response_tokens: list[int] = []
        print("\nAssistant: ", end="", flush=True)
        for token_column, _token_masks in engine.generate(
            conversation_tokens,
            num_samples=1,
            max_tokens=256,
            temperature=args.temperature,
            top_k=args.top_k,
        ):
            token: int = token_column[0]
            response_tokens.append(token)
            print(tokenizer.decode([token]), end="", flush=True)
        print()

        if not response_tokens or response_tokens[-1] != assistant_end:
            response_tokens.append(assistant_end)
        conversation_tokens.extend(response_tokens)

        if args.prompt:
            break


if __name__ == "__main__":
    main()
