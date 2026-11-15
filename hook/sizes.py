"""How big a chat really is, in real tokens (Autoswitch v2, used only when AUTOSWITCH_HANDOFF is on).

Step 1 measured real tokens = characters / 4 x 1.6, one factor for Haiku, Sonnet and Opus
(AUTOSWITCH_TOKEN_FACTOR). The old estimate, transcript bytes / 4, runs about 5x too high on a typical chat
and up to 100x on chats with big tool logs, because the transcript also stores API request copies, metadata,
file snapshots and base64 images.

Two paths:
  cheap    file bytes / 4 / AUTOSWITCH_FILE_OVERCOUNT (5): free, only used for small files.
  precise  the conversation text itself: message text, tool calls and tool results of the main chat
           (sidechains, thinking, metadata and API request copies left out), restarting at each compaction
           boundary, plus a fixed AUTOSWITCH_IMAGE_TOKENS (1500) per image. A 64 MB transcript takes
           about 0.15 s on an M-series Mac.
The precise count runs when the file is at least AUTOSWITCH_PRECISE_BYTES (120 KB), or when asked for
(the zone around break-even). Why 120 KB: a file can never hold more real tokens than bytes x 1.6 / 4
(every byte a character), so below 120 KB a chat is under 48k real tokens, below every break-even zone, and
the cheap number is only used for the in-place cost check. The 5x overcount does not hold for every chat
(a chat of long pasted text is close to 1x), so the cheap number never decides near break-even.
Both numbers are returned so shadow.jsonl can log them side by side.

When the chat has a reply with API usage, estimate() uses it instead (method "exact"): the messages are the
exact size the API recorded on the last reply minus the chat's starting size (its first reply: the tools and
system prompt, about 78k, that every chat carries and that a handoff pays too). Step 1's break-evens (55k,
70k) were measured on message size, and the precise count reads 30-35% low against the API, so a chat used to
switch in place until it really had 75-100k of messages.
"""
import json
import os

from policy import _f


def factor():
    return _f("AUTOSWITCH_TOKEN_FACTOR", "1.6")


def cheap(path):
    """Real tokens guessed from the file size, or None if the file can't be read."""
    try:
        size = os.path.getsize(path)
    except (OSError, TypeError, ValueError):
        return None
    return size / 4.0 / max(_f("AUTOSWITCH_FILE_OVERCOUNT", "5"), 0.01)


def _result_chars(content):
    if isinstance(content, str):
        return len(content), 0
    chars = imgs = 0
    for x in content or []:
        if not isinstance(x, dict):
            continue
        if x.get("type") == "text":
            chars += len(x.get("text") or "")
        elif x.get("type") == "image":
            imgs += 1
    return chars, imgs


def _typed(c):
    """A user entry that is a message (text), not a tool result."""
    if isinstance(c, str):
        return True
    blocks = [b for b in c or [] if isinstance(b, dict)]
    return any(b.get("type") == "text" for b in blocks) and not any(b.get("type") == "tool_result" for b in blocks)


def text_chars(path, marks=None):
    """(characters, images) of the conversation the model re-reads, or None if unreadable.
    marks: if a list, gets (characters, images) so far at the start of each typed user message (rollover's
    growth rate); cleared at a compaction like the count."""
    chars = imgs = 0
    try:
        with open(path, "rb") as f:
            for line in f:
                if b'"compact_boundary"' in line and b'"system"' in line:
                    try:
                        if json.loads(line).get("subtype") == "compact_boundary":
                            chars = imgs = 0  # the model only re-reads what follows the compaction
                            if marks is not None:
                                del marks[:]
                    except ValueError:
                        pass
                    continue
                if b'"user"' not in line and b'"assistant"' not in line:
                    continue  # api-request copies, snapshots, metadata: never re-read by the model
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(d, dict) or d.get("type") not in ("user", "assistant") or d.get("isSidechain"):
                    continue
                c = (d.get("message") or {}).get("content") if isinstance(d.get("message"), dict) else None
                if marks is not None and d.get("type") == "user" and _typed(c):
                    marks.append((chars, imgs))
                if isinstance(c, str):
                    chars += len(c)
                    continue
                for b in c or []:
                    if not isinstance(b, dict):
                        continue
                    t = b.get("type")
                    if t == "text":
                        chars += len(b.get("text") or "")
                    elif t == "tool_use":
                        chars += len(json.dumps(b.get("input"))) + len(b.get("name") or "")
                    elif t == "tool_result":
                        n, i = _result_chars(b.get("content"))
                        chars += n
                        imgs += i
                    elif t == "image":
                        imgs += 1
    except (OSError, TypeError, ValueError):
        return None
    return chars, imgs


def precise(path):
    """Real tokens counted from the conversation text, or None if unreadable."""
    r = text_chars(path)
    if r is None:
        return None
    chars, imgs = r
    return chars / 4.0 * factor() + imgs * _f("AUTOSWITCH_IMAGE_TOKENS", "1500")


def exact_messages(path):
    """(message tokens, total, start) from the API usage, or None if no reply has usage yet. start is the
    first reply's size (tools, system prompt and the first message), capped at AUTOSWITCH_START_TOKENS_MAX
    (100k) so a chat that opens with a huge paste still counts most of it as messages."""
    seen = []
    total = exact(path, first=seen)
    if total is None or not seen:
        return None
    start = min(seen[0], _f("AUTOSWITCH_START_TOKENS_MAX", "100000"))
    return max(0.0, float(total) - start), float(total), float(start)


def estimate(path, force_precise=False):
    """{"tokens", "cheap", "precise", "method"}: tokens is the number to decide with (message tokens), None if
    unknown. method: "exact" | "cheap" | "precise" | "unknown"; "exact" also gives "exact_total" and "start"."""
    c = cheap(path)
    if c is None:
        return {"tokens": None, "cheap": None, "precise": None, "method": "unknown"}
    if os.environ.get("AUTOSWITCH_EXACT_SIZE", "on").strip().lower() != "off":
        e = exact_messages(path)
        if e is not None:
            return {"tokens": e[0], "cheap": c, "precise": None, "method": "exact", "exact_total": e[1],
                    "start": e[2]}
    p = None
    try:
        big = os.path.getsize(path) >= _f("AUTOSWITCH_PRECISE_BYTES", "120000")
    except OSError:
        big = False
    if big or force_precise:
        p = precise(path)
    if p is not None:
        return {"tokens": p, "cheap": c, "precise": p, "method": "precise"}
    return {"tokens": c, "cheap": c, "precise": None, "method": "cheap"}


def exact(path, marks=None, first=None):
    """The chat's real context size in tokens, from the API usage the app records on every reply (input +
    cache read + cache write: what the model actually re-reads, tools and system prompt included, and what the
    app's context meter shows). None if no reply has usage yet. marks: if a list, gets the size at the start of
    each typed user message (the last reply's size before it); cleared at a compaction. first: if a list, gets
    the first reply's size (the chat's starting size; not reset by a compaction)."""
    last = None
    try:
        with open(path, "rb") as f:
            for line in f:
                if b'"compact_boundary"' in line and b'"system"' in line:
                    try:
                        if json.loads(line).get("subtype") == "compact_boundary" and marks is not None:
                            del marks[:]
                    except ValueError:
                        pass
                    continue
                if b'"usage"' not in line and not (marks is not None and b'"user"' in line):
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(d, dict) or d.get("isSidechain"):
                    continue
                msg = d.get("message") if isinstance(d.get("message"), dict) else {}
                if d.get("type") == "assistant":
                    u = msg.get("usage")
                    if isinstance(u, dict):
                        n = sum(int(u.get(k) or 0) for k in
                                ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
                        if n > 0:
                            last = n
                            if first is not None and not first:
                                first.append(n)
                elif d.get("type") == "user" and marks is not None and last is not None and _typed(msg.get("content")):
                    marks.append(last)
    except (OSError, TypeError, ValueError):
        return None
    return last
