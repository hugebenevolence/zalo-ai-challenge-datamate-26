"""Observe turn for small models: a sample whose query ran gets to see what came back, then either keeps it or corrects it.
The harness does the checking (shape, first rows, filter values the shop does not hold), so the model needs no exploration tools
and no free loop: one fixed extra turn. A correction replaces the sample only if it runs; the original is never lost."""
import json
import re
from difflib import get_close_matches

from ingest import norm, quote


def known_values(db, max_distinct=50000):
    """norm(value) -> value for every text value of the shop, canonical tables included."""
    known = {}
    for tab, col in db.execute("SELECT table_name, column_name FROM information_schema.columns WHERE data_type = 'VARCHAR' AND table_name <> 'calendar' "
                               "AND NOT starts_with(table_name, '_')").fetchall():
        for (v,) in db.execute(f"SELECT DISTINCT {quote(col)} FROM {quote(tab)} WHERE {quote(col)} IS NOT NULL LIMIT {max_distinct}").fetchall():
            known.setdefault(norm(v), str(v))
    return known


def missing_filters(sql, known):
    """Equality filter values that no column of the shop holds, with the closest spelling that exists."""
    out = []
    for lit in dict.fromkeys(re.findall(r"(?<![<>!])=\s*'([^']+)'", sql)):
        k = norm(lit)
        if k and not k.isdigit() and k not in known:
            near = get_close_matches(k, known, n=1, cutoff=0.75)
            out.append((lit, known[near[0]] if near else None))
    return out


def observation(sql, names, rows, known):
    """What the model sees after a query that ran."""
    head = json.dumps(rows[:5], ensure_ascii=False, default=str)
    text = f"Your query ran and returned {len(rows)} row(s) with columns {json.dumps(names, ensure_ascii=False)}.\nFirst rows: {head[:600]}"
    if any(r and r[0] is None for r in rows[:5]) or (len(rows) == 1 and all(v in (0, None) for v in rows[0])):
        text += "\nWarning: the result is 0 or NULL. A filter may match nothing."
    for lit, near in missing_filters(sql, known):
        text += f"\nWarning: the filter value '{lit}' is in no column of this shop." + (f" Closest value: '{near}'." if near else "")
    return (text + "\nCheck it against the question: the objects asked for, the number of rows, the unit and the period. If it answers the question, reply FINAL. "
            "If not, write the corrected TYPE line and one ```sql``` block.")


def observe_turn(generate, execute, run_sql, parse, db, known, samples):
    """samples: {key: (conversation up to the sample's reply, text, pred, question)}. generate(conversations) -> reply texts.
    Returns {key: {"reply", "before", "revised"}} and the revised (text, pred) per key."""
    keys, convs = [], []
    for key, (conv, text, pred, _) in samples.items():
        _, sql = parse(text.split("\n#REPAIR\n")[-1])
        if not sql or pred is None or pred.get("type") == "unanswerable":
            continue
        try:
            names, rows = run_sql(db, sql.rstrip().rstrip(";"))
        except Exception:
            continue
        keys.append(key)
        convs.append(conv + [{"role": "user", "content": observation(sql, names, rows, known)}])
    log, revised = {}, {}
    for key, reply in zip(keys, generate(convs) if keys else []):
        _, text, pred, question = samples[key]
        log[key] = {"reply": reply[:1500], "before": pred, "revised": False}
        if parse(reply)[1]:
            try:
                new = execute(db, reply, question)
            except Exception as e:
                log[key]["failed"] = str(e)[:200]
                continue
            if new != pred:
                log[key]["revised"] = True
                revised[key] = (text + "\n#REPAIR\n" + reply, new)
    return log, revised
