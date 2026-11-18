"""Handoff vs compaction vs re-reading, across many chats, sizes and incoming models.

Runs on the Claude subscription through Claude Code (Agent SDK), never an API key: it
refuses to start a call that would bill a key.

For each point (one chat cut at one size):
  1. Opus writes the question set from the full history: the same four categories for
     every chat (decisions, early instructions, file and code state, what's left to do).
  2. Opus (the outgoing model, as in Claude Code) writes a compaction summary with the
     compaction prompt of the earlier benchmark.
  3. The ACPMH handoff is built exactly as Autoswitch ships it: word-for-word journal, no
     API-billed fact extraction; balanced mode (v4 under 300k real tokens, v5b above).
  4. Each incoming model (Haiku, Sonnet, Opus) answers from: the whole history (re-read),
     the compaction summary, and the handoff (with the journal lookup tools, as part 2 has).
  5. Opus grades every answer against the gold answer, blind to the method.

Every stage is cached in the point's folder, so a rerun only does what is missing.
The chats themselves are private: list them in a chats file that is not published, e.g.
  [{"id": "router-coding", "kind": "coding", "transcript": "/path/to/chat.jsonl"}]

Usage: uv run --with claude-agent-sdk==0.2.182 python3 bench/handoff_vs_compaction.py \
           --chats bench/chats.json --sizes 200000 --out bench/out/multi
Summary across points: python3 bench/handoff_vs_compaction.py --report --out bench/out/multi
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time

for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
    os.environ.pop(k, None)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from acpmh import handoff, ingest, tools  # noqa: E402
from acpmh.store import Store, est_tokens  # noqa: E402

# Prompts from the earlier one-chat benchmark (docs/BENCHMARK.md), unchanged.
P = {
    'COMPACT': 'Your task is to create a detailed summary of the conversation so far, so that\nwork can continue without losing context. It will replace the conversation. Include:\n1. Primary request and intent (all explicit user requests)\n2. Key technical concepts\n3. Files and code sections (with important snippets)\n4. Errors and fixes (including user feedback)\n5. Problem solving done\n6. All user messages that are not tool results\n7. Pending tasks\n8. Current work, in detail, right before this summary\n9. Optional next step\nBe thorough and precise; keep exact names, numbers and paths.',
    'GRADE': 'Grade answers from an AI model against gold answers for questions about a\ncoding project. For each question: 1 = correct and complete (all key specifics right),\n0.5 = partially correct or missing a key specific, 0 = wrong, "unknown", or missing.\nWording does not matter; specifics do. Reply with JSON only:\n{"grades":[{"id":1,"score":1,"why":"short reason"}]}\n\n',
    'ANSWER_RULES': 'Answer each question as precisely as you can (exact names, numbers, paths).\nIf you do not know, say "unknown" rather than guessing.\nReply with JSON only: {"answers":[{"id":1,"answer":"..."}]}',
}

MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
OPUS = MODELS["opus"]
REAL_PER_EST = 1.6  # real tokens per est_tokens unit (measured in RESULTS50.md, both tokenizers)
V4 = {"full_log": True, "tail_tokens": 8000}
V5B = {"user_log": True, "tail_tokens": 20000, "max_tokens": 40000}
CATEGORIES = ("decisions", "early", "files", "todo")

QGEN = """Below is the complete history of a Claude Code chat, up to the exact moment a
different AI model will take over. Write {n} questions that test whether the incoming model
knows what it needs to continue correctly, exactly {k} in each category:
- "decisions": decisions made and why, including decisions that were later reversed (ask
  for the decision in force) and approaches rejected and why;
- "early": instructions, requirements or preferences the user gave in the EARLY part of
  the chat that still apply;
- "files": the current state of files and code: which files exist or changed, what a
  specific file or function now does, exact names, paths, values, commands;
- "todo": what is left to do: the current task, what is pending or blocked, and the
  immediate next step.
Each question must have one clear answer fully supported by the history; ask about the
LATEST state when something changed. Do not ask about wording or tool-call mechanics.
Give a concise but complete gold answer and a short verbatim evidence quote.

Reply with JSON only:
{{"questions":[{{"id":1,"category":"decisions|early|files|todo","question":"...","gold":"...","evidence":"..."}}]}}

HISTORY:
{history}"""

INTRO = {"full": "The full history of its session follows.",
         "compact": "A summary of its session follows.",
         "handoff": "The ACPMH handoff for it follows. Use the journal tools whenever the "
                    "handoff does not state an answer exactly."}
# handoff_lookup: the same handoff plus a firm lookup rule. Weaker models (Haiku, Sonnet) answered
# "unknown" without ever searching the journal; this tests the rule the product handoff would carry.
LOOKUP_RULE = ("The handoff does not contain everything; the journal does. Whenever you need a detail "
               "the handoff does not state word for word (an earlier instruction, a decision and its "
               "reason, an exact name, value, path or file content, what is left to do), search the "
               "journal with search_history and read the events with get_event before you answer. "
               "Never answer \"unknown\" until you have searched for it.")
INTRO["handoff_lookup"] = INTRO["handoff"] + " " + LOOKUP_RULE


def jload(p):
    with open(p) as f:
        return json.load(f)


def jsave(p, obj):
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, p)


def parse_obj(text):
    i, j = text.find("{"), text.rfind("}")
    return json.loads(text[i:j + 1])


def ask(prompt, model, system, log, purpose, mcp=None, max_turns=1):
    """One Claude Code turn on the subscription. Returns the final text."""
    import anyio
    from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage,
                                  SystemMessage, TextBlock, query)
    out = {"texts": [], "res": None}
    kw = dict(model=model, system_prompt=system, tools=[], allowed_tools=[], setting_sources=[],
              max_turns=max_turns, effort="medium",
              env={"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""})
    if mcp:
        kw["mcp_servers"] = {"acpmh": mcp}
        kw["allowed_tools"] = ["mcp__acpmh__" + t["name"] for t in tools.TOOLS
                               if t["name"] != "get_image"]

    async def go():
        async for m in query(prompt=prompt, options=ClaudeAgentOptions(**kw)):
            if isinstance(m, SystemMessage) and m.subtype == "init":
                if m.data.get("apiKeySource") not in (None, "none"):
                    raise RuntimeError("Refusing: this would bill an API key, not the subscription.")
            elif isinstance(m, AssistantMessage):
                t = "".join(b.text for b in m.content if isinstance(b, TextBlock))
                if t:
                    out["texts"].append(t)
            elif isinstance(m, ResultMessage):
                out["res"] = m

    t0 = time.time()
    anyio.run(go)
    r = out["res"]
    rec = {"purpose": purpose, "model": model, "seconds": round(time.time() - t0),
           "billing": "subscription"}
    if r is not None:
        rec.update(usage=r.usage, usd_equivalent=r.total_cost_usd, turns=r.num_turns)
    log.append(rec)
    if r is None or r.is_error:
        raise RuntimeError("%s failed: %s" % (purpose, getattr(r, "result", "no result")))
    print("    %s (%s) %ds, ~$%.2f equiv" % (purpose, model.split("-")[1], rec["seconds"],
                                             rec.get("usd_equivalent") or 0), flush=True)
    return r.result or (out["texts"][-1] if out["texts"] else "")


def journal_server(store):
    """The ACPMH journal tools, served in-process to the incoming model."""
    from claude_agent_sdk import create_sdk_mcp_server, tool
    fns = []
    for t in tools.TOOLS:
        if t["name"] == "get_image":
            continue

        def make(name):
            async def fn(args):
                res = tools.call(store, name, args)
                return {"content": [{"type": "text", "text": res if isinstance(res, str)
                                     else json.dumps(res, default=str)}]}
            return fn
        fns.append(tool(t["name"], t["description"], t["input_schema"])(make(t["name"])))
    return create_sdk_mcp_server(name="acpmh", version="1.0", tools=fns)


def point(chat, size, out_root, models, n_questions, arms=("full", "compact", "handoff")):
    entries = ingest.load(chat["transcript"])
    # "1M" is cut at 920k: a full 1M history plus instructions does not fit a 1M window.
    target = int(min(size, 920000) / REAL_PER_EST)
    cut = ingest.cut_index(entries, target)
    prefix = entries[:cut]
    history = "\n\n".join(t for _, t in ingest.render(prefix))
    est = est_tokens(history)
    if est < target * 0.95:
        print("  %s is only ~%dk real tokens, skipping %dk" % (chat["id"], est * REAL_PER_EST / 1000,
                                                             size / 1000))
        return
    d = os.path.join(out_root, "%s_%dk" % (chat["id"], size // 1000))
    os.makedirs(d, exist_ok=True)
    meta = {"chat": chat["id"], "kind": chat.get("kind"), "size": size, "entries": cut,
            "of_entries": len(entries), "history_chars": len(history),
            "history_real_tokens_est": int(est * REAL_PER_EST)}
    jsave(os.path.join(d, "meta.json"), meta)
    print("== %s at %dk (cut at entry %d of %d)" % (chat["id"], size // 1000, cut, len(entries)),
          flush=True)

    qpath = os.path.join(d, "questions.json")
    if not os.path.exists(qpath):
        log = []
        k = n_questions // len(CATEGORIES)
        qs = parse_obj(ask(QGEN.format(n=k * len(CATEGORIES), k=k, history=history), OPUS,
                           "You write evaluation questions.", log, "questions"))["questions"]
        jsave(qpath, qs)
        jsave(os.path.join(d, "usage_questions.json"), log)
    qs = jload(qpath)
    qblock = "\n".join("%d. %s" % (q["id"], q["question"]) for q in qs)

    spath = os.path.join(d, "compact_summary.md")
    if not os.path.exists(spath):
        log = []
        s = ask(history + "\n\n" + P["COMPACT"], OPUS, "You summarise AI coding sessions.", log,
                "compaction")
        open(spath, "w").write(s)
        jsave(os.path.join(d, "usage_compaction.json"), log)
    summary = open(spath).read()

    store = Store(os.path.join(d, "store"))
    if store.last_seq() == 0:
        ingest.ingest(store, prefix)
    hpath = os.path.join(d, "handoff.md")
    variant = "v4" if size < 300000 else "v5b"
    if not os.path.exists(hpath):
        text, stats = handoff.build(store, **(V4 if variant == "v4" else V5B))
        open(hpath, "w").write(text)
        jsave(os.path.join(d, "handoff_stats.json"), dict(stats, variant=variant))
    htext = open(hpath).read()

    material = {"full": "HISTORY:\n" + history, "compact": "SUMMARY:\n" + summary,
                "handoff": htext}
    for mname in models:
        model = MODELS[mname]
        for arm in arms:
            rpath = os.path.join(d, "result_%s_%s.json" % (arm, mname))
            if os.path.exists(rpath):
                continue
            log = []
            sysmsg = "You are taking over a software project from another AI model. " + INTRO[arm]
            if arm.startswith("handoff"):
                sysmsg += "\n\n" + htext
                prompt = "QUESTIONS:\n" + qblock + "\n\n" + P["ANSWER_RULES"]
            else:
                prompt = material[arm] + "\n\nQUESTIONS:\n" + qblock + "\n\n" + P["ANSWER_RULES"]
            try:
                text = ask(prompt, model, sysmsg, log, "%s/%s" % (arm, mname),
                           mcp=journal_server(store) if arm.startswith("handoff") else None,
                           max_turns=(30 if arm == "handoff_lookup" else 16) if arm.startswith("handoff") else 1)
                answers = parse_obj(text)["answers"]
            except Exception as e:  # e.g. the history does not fit the model's window
                msg = str(e).lower()
                if not any(w in msg for w in ("too long", "context", "exceed", "too many tokens")):
                    raise  # usage limit, network...: stop, the runner retries later
                jsave(rpath, {"arm": arm, "model": mname, "error": str(e)[:500], "usage": log})
                print("    %s/%s did not run: %s" % (arm, mname, str(e)[:160]), flush=True)
                continue
            by_id = {a["id"]: a.get("answer", "") for a in answers}
            body = "\n\n".join("Q%d: %s\nGOLD: %s\nANSWER: %s" % (
                q["id"], q["question"], q["gold"], by_id.get(q["id"], "(missing)")) for q in qs)
            glog = []
            grades = parse_obj(ask(P["GRADE"] + body, OPUS, "You grade answers.", glog,
                                   "grade %s/%s" % (arm, mname)))["grades"]
            cat = {q["id"]: q.get("category") for q in qs}
            by_cat = {}
            for g in grades:
                by_cat.setdefault(cat.get(g["id"]), []).append(float(g["score"]))
            score = sum(float(g["score"]) for g in grades)
            jsave(rpath, {"arm": arm, "model": mname, "score": score, "max": len(qs),
                          "by_category": {c: sum(v) for c, v in by_cat.items()},
                          "answers": answers, "grades": grades, "usage": log,
                          "grading_usage": glog})
            print("    %s/%s: %.1f/%d" % (arm, mname, score, len(qs)), flush=True)


def usd(log):
    return sum(r.get("usd_equivalent") or 0 for r in log)


def report(out_root):
    """Per point and per incoming model: score as % of re-reading, usage per switch.
    Then the spread across chats for each size, model and method."""
    rows = []
    for d in sorted(glob.glob(os.path.join(out_root, "*_*k"))):
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        meta = jload(os.path.join(d, "meta.json"))
        one_time = {}
        for name, key in (("usage_compaction.json", "compact"),):
            p = os.path.join(d, name)
            if os.path.exists(p):
                one_time[key] = usd(jload(p))
        res = {}
        for p in glob.glob(os.path.join(d, "result_*.json")):
            r = jload(p)
            res[(r["arm"], r["model"])] = r
        ref = res.get(("full", "opus")) or res.get(("full", "sonnet"))  # reference re-read
        for (arm, model), r in sorted(res.items()):
            if "error" in r:
                rows.append(dict(meta, arm=arm, model=model, error=r["error"][:80]))
                continue
            full = res.get(("full", model))
            base = full if full and "error" not in full else ref
            cost = usd(r["usage"]) + one_time.get(arm, 0)
            rows.append(dict(meta, arm=arm, model=model, score=r["score"], max=r["max"],
                             pct_of_reread=100.0 * r["score"] / base["score"] if base and base.get("score") else None,
                             usd_per_switch=round(cost, 3), by_category=r.get("by_category")))
    jsave(os.path.join(out_root, "summary.json"), rows)
    print("%-22s %-6s %-7s %-7s %6s %8s" % ("point", "model", "method", "score", "%rr", "usage$"))
    for r in rows:
        if "error" in r:
            print("%-22s %-6s %-7s did not run (%s)" % ("%s@%dk" % (r["chat"], r["size"] // 1000),
                                                        r["model"], r["arm"], r["error"][:40]))
            continue
        print("%-22s %-6s %-7s %4.1f/%-2d %5.0f%% %8.2f" % (
            "%s@%dk" % (r["chat"], r["size"] // 1000), r["model"], r["arm"], r["score"], r["max"],
            r["pct_of_reread"] or 0, r["usd_per_switch"]))
    print("\nAcross chats (median, min-max):")
    groups = {}
    for r in rows:
        if "error" not in r and r["pct_of_reread"] is not None:
            groups.setdefault((r["size"], r["model"], r["arm"]), []).append(r)
    for (size, model, arm), rs in sorted(groups.items()):
        pc = [r["pct_of_reread"] for r in rs]
        us = [r["usd_per_switch"] for r in rs]
        print("  %4dk %-6s %-7s n=%d  remembers %3.0f%% (%.0f-%.0f%%)  usage $%.2f ($%.2f-%.2f)" % (
            size // 1000, model, arm, len(rs), statistics.median(pc), min(pc), max(pc),
            statistics.median(us), min(us), max(us)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chats")
    ap.add_argument("--only", help="comma-separated chat ids")
    ap.add_argument("--sizes", default="200000")
    ap.add_argument("--models", default="haiku,sonnet,opus")
    ap.add_argument("--arms", default="full,compact,handoff")
    ap.add_argument("--questions", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report(a.out)
    chats = jload(a.chats)
    if a.only:
        keep = set(a.only.split(","))
        chats = [c for c in chats if c["id"] in keep]
    for size in (int(s) for s in a.sizes.split(",")):
        for c in chats:
            point(c, size, a.out, a.models.split(","), a.questions, a.arms.split(","))
    report(a.out)


if __name__ == "__main__":
    main()
