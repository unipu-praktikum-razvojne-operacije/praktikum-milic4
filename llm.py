"""Command line interface for the local Ollama server."""
import argparse
import sys

import ollama_client as oc


def cmd_status(args):
    if oc.is_running():
        print("Ollama is running")
        return 0
    print("Ollama is NOT running (start it with: ollama serve)")
    return 1


def cmd_models(args):
    for name in oc.list_models():
        print(name)
    return 0


def cmd_ask(args):
    print(oc.generate(args.question, model=args.model, system=args.system))
    return 0


def cmd_chat(args):
    history = []
    print("Interactive chat. Type 'exit' to quit.")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        messages = oc.build_messages(prompt, system=args.system, history=history)
        reply = oc.chat(messages, model=args.model)
        print(f"llm> {reply}")
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": reply})
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="CLI for a local Ollama model")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="check if the Ollama server is running").set_defaults(func=cmd_status)
    sub.add_parser("models", help="list downloaded models").set_defaults(func=cmd_models)

    ask = sub.add_parser("ask", help="ask a single question")
    ask.add_argument("question")
    ask.add_argument("--system", help="system prompt")
    ask.add_argument("--model", help="model name")
    ask.set_defaults(func=cmd_ask)

    chat = sub.add_parser("chat", help="interactive chat with history")
    chat.add_argument("--system", help="system prompt")
    chat.add_argument("--model", help="model name")
    chat.set_defaults(func=cmd_chat)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except oc.OllamaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
