"""Old parts stay quiet when only their own next part reports in.

A part that continues in a new chat (rollover or handoff) starts that chat with start_session, so the app links
them: each time the new part ends a turn, the app wakes the old part with a <side-sessions-event> note. The old
part then re-reads its whole chat (often 200k-400k tokens) just to say "work continues in part N", and its own
turn ending wakes the part before it, so a chain of parts wakes all the way up. Detaching the new part stops this,
but the app refuses to detach a session that shares its parent's checkout (every part of a chat in a worktree).

So the prompt hook drops such a turn before any model call: when every session the note is about is this chat's
own next part. A note that also mentions another session, any other kind of event, and anything the user types go
through as before.
"""
import re

WOKE_BY = re.compile(r"you started \((local_[0-9A-Za-z_-]+)\) finished a turn")
EVENT = "<side-sessions-event"


def successors(app_id):
    """App ids of the chats that took over this chat's work (rollover or handoff)."""
    out = set()
    if not app_id:
        return out
    try:
        import rollover
        for r in rollover.all_records():
            if r.get("session_id") == app_id and r.get("status") in rollover.CONTINUED and r.get("part2_session"):
                out.add(r["part2_session"])
    except Exception:
        pass
    try:
        import handoff
        for r in handoff.all_records():
            if r.get("session_id") == app_id and r.get("status") == "delivered" and r.get("part2_session"):
                out.add(r["part2_session"])
    except Exception:
        pass
    return out


def only_successor_news(raw, app_id):
    """True when this machine turn is just the app relaying this chat's next part(s) finishing a turn."""
    if not raw or EVENT not in raw:
        return False
    ids = set(WOKE_BY.findall(raw))
    if not ids:
        return False
    succ = successors(app_id)
    return bool(succ) and ids <= succ


def hook_output(raw, app_id):
    """The UserPromptSubmit output that drops the turn, or None to let it through."""
    if not only_successor_news(raw, app_id):
        return None
    return {"continue": False,
            "stopReason": "↪ The next part finished a turn; this earlier part stays idle (nothing re-read)."}
