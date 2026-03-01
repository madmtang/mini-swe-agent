from __future__ import annotations

import re
import shlex
import textwrap
from collections.abc import Iterable

try:
    import bashlex
except ModuleNotFoundError:
    bashlex = None


# Common wrapper commands that often "execute another command" as their argument.
# We include both the wrapper and the wrapped command (when we can infer it).
_WRAPPERS: set[str] = {
    "sudo",
    "env",
    "command",
    "builtin",
    "time",
    "nice",
    "nohup",
    "chronic",
    "stdbuf",
    "setpriv",
    "chpst",
    "gosu",
    "su",
    "chroot",
    "ionice",
    "taskset",
}

_COMMAND_SEPARATORS = {"|", "||", "&", "&&", ";", ";;", "(", ")"}
_REDIRECTION_TOKENS = {"<", ">", "<<", ">>", "<<<", "<>", ">&", "<&", ">|", "<<-"}
_NON_COMMAND_TOKENS = {"$", "!", "{"}


def extract_executables(cmdline: str) -> list[str]:
    """
    Parse a bash script/command line using bashlex and return a list of
    command words (executables/builtins) in source order.

    Notes:
      - This is a syntactic extraction, not semantic evaluation.
      - It includes commands appearing in conditionals, loops, functions,
        subshells, and command substitutions (when bashlex exposes them).
      - For wrappers like 'sudo'/'env', it will usually include both the wrapper
        and the wrapped command (when inferable).
    """
    if bashlex is None:
        return _extract_executables_without_bashlex(cmdline)

    try:
        trees = bashlex.parse(cmdline)
    except Exception:
        # Fallback for common bashlex heredoc edge cases (e.g. <<'EOF').
        try:
            trees = bashlex.parse(_bashlex_fallback_source(cmdline))
        except Exception:
            # If parsing still fails, return empty rather than crashing.
            return []

    out: list[str] = []
    for t in trees:
        _walk(t, out)
    return out


def _walk(node: object, out: list[str]) -> None:
    if node is None:
        return

    # bashlex AST nodes are typically instances of bashlex.ast.node
    kind = getattr(node, "kind", None)

    if kind == "command":
        _handle_command_node(node, out)

    # Recurse into children (generic reflection over node fields).
    for child in _iter_children(node):
        _walk(child, out)


def _handle_command_node(cmd_node: object, out: list[str]) -> None:
    parts = getattr(cmd_node, "parts", None)
    if not isinstance(parts, list):
        return

    # Collect word-like tokens in this simple command, in order.
    words: list[str] = []
    for p in parts:
        pk = getattr(p, "kind", None)
        if pk == "word":
            w = _word_text(p)
            if w is not None:
                words.append(w)

    _append_command_words(words, out)


def _append_command_words(words: list[str], out: list[str]) -> None:
    if not words:
        return

    # Identify the "command name" position:
    # skip leading assignments like FOO=bar (syntactic heuristic).
    i = 0
    while i < len(words) and _looks_like_assignment_word(words[i]):
        i += 1
    if i >= len(words):
        return

    cmd0 = words[i]
    out.append(cmd0)

    # If it is a wrapper, try to also infer the wrapped command.
    if cmd0 in _WRAPPERS:
        wrapped = _infer_wrapped_command(cmd0, words[i + 1 :])
        if wrapped is not None:
            out.append(wrapped)


def _infer_wrapped_command(wrapper: str, args: list[str]) -> str | None:
    """
    Very lightweight inference of the next command after some common wrappers.
    """
    if not args:
        return None

    if wrapper == "env":
        # env [-i] [-u NAME] [NAME=VALUE]... COMMAND [ARG]...
        j = 0
        while j < len(args):
            a = args[j]
            if a == "--":
                j += 1
                break
            if a.startswith("-"):
                # -u NAME consumes the next token
                if a in {"-u", "--unset"} and j + 1 < len(args):
                    j += 2
                    continue
                j += 1
                continue
            if _looks_like_assignment_word(a):
                j += 1
                continue
            return a
        if j < len(args):
            # After "--"
            while j < len(args) and _looks_like_assignment_word(args[j]):
                j += 1
            if j < len(args):
                return args[j]
        return None

    if wrapper == "sudo":
        # sudo [options] -- COMMAND ...
        j = 0
        while j < len(args):
            a = args[j]
            if a == "--":
                j += 1
                break
            if a.startswith("-"):
                # Some sudo options consume an argument; we do not try to model all.
                # Heuristic: skip the option token; if it is exactly "-u" or "-g",
                # also skip the next token if present.
                if a in {"-u", "-g", "-h", "-p", "-U", "-t", "-C"} and j + 1 < len(args):
                    j += 2
                else:
                    j += 1
                continue
            return a
        if j < len(args):
            return args[j]
        return None

    if wrapper in {"command", "builtin"}:
        # command [-pVv] COMMAND ...
        j = 0
        while j < len(args) and args[j].startswith("-"):
            j += 1
        return args[j] if j < len(args) else None

    if wrapper in {
        "time",
        "nice",
        "nohup",
        "chronic",
        "stdbuf",
        "setpriv",
        "chpst",
        "gosu",
        "su",
        "chroot",
        "ionice",
        "taskset",
    }:
        # Many of these have options; just skip leading '-' options until the first non-option.
        j = 0
        while j < len(args):
            a = args[j]
            if a == "--":
                j += 1
                break
            if a.startswith("-"):
                j += 1
                continue
            return a
        return args[j] if j < len(args) else None

    return None


def _looks_like_assignment_word(s: str) -> bool:
    # Minimal syntactic heuristic: NAME=VALUE with a valid-ish NAME.
    if "=" not in s:
        return False
    name, _ = s.split("=", 1)
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(ch.isalnum() or ch == "_" for ch in name)


def _word_text(word_node: object) -> str | None:
    # bashlex "word" node usually has .word
    w = getattr(word_node, "word", None)
    if isinstance(w, str):
        return w
    # Fallbacks (rare, but keep robust)
    v = getattr(word_node, "value", None)
    if isinstance(v, str):
        return v
    return None


def _bashlex_fallback_source(src: str) -> str:
    # bashlex can fail on quoted heredoc delimiters like <<'EOF'; normalize them.
    normalized = re.sub(
        r"""(<<-?\s*)(['"])([A-Za-z_][A-Za-z0-9_]*)\2""",
        r"\1\3",
        src,
    )
    return textwrap.dedent(normalized)


def _extract_executables_without_bashlex(cmdline: str) -> list[str]:
    """Best-effort shell parsing when bashlex is unavailable."""
    try:
        lexer = shlex.shlex(cmdline, posix=True, punctuation_chars="|&;()<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return []

    out: list[str] = []
    words: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _COMMAND_SEPARATORS:
            _append_command_words(words, out)
            words = []
            i += 1
            continue
        if token in _REDIRECTION_TOKENS:
            i += 2
            continue
        if token in _NON_COMMAND_TOKENS:
            i += 1
            continue
        words.append(token)
        i += 1

    _append_command_words(words, out)
    return out


def _iter_children(node: object) -> Iterable[object]:
    """
    Generic AST traversal by reflecting over node.__dict__.

    This lets us reach nested constructs like:
      - pipelines, lists, conditionals, loops
      - function bodies
      - command substitutions embedded in words (when represented as nodes)
    """
    d = getattr(node, "__dict__", None)
    if not isinstance(d, dict):
        return []

    children: list[object] = []
    for _, v in d.items():
        if v is None:
            continue
        if _is_ast_node(v):
            children.append(v)
        elif isinstance(v, (list, tuple)):
            for x in v:
                if _is_ast_node(x):
                    children.append(x)
    return children


def _is_ast_node(x: object) -> bool:
    # bashlex nodes typically have a .kind attribute
    return hasattr(x, "kind")


__all__ = ["extract_executables"]
