"""Agent steps before the SQL samples (--agent):
UNDERSTAND: the LLM reads the question once (greedy) and writes JSON: the values to look up, the time phrases, the
rulebook rules the answer depends on and the answer shape. It answers nothing.
GROUND: code looks every value up in the shop's text columns (accents, case, spacing and small typos ignored) and turns
every time phrase into its rulebook date window with the row count in it.
Only the grounded facts and the rule rows go into the SQL prompt; the reading itself does not, so a misreading cannot
make all K samples agree on it."""
import json
import re
from datetime import date, timedelta
from difflib import SequenceMatcher

from ingest import ckey, norm, quote

UNDERSTAND_SYSTEM = """You read ONE Vietnamese question about a shop's data. It may use slang, abbreviations, no diacritics or typos.
Do not answer it and do not write SQL: write what it asks as JSON, in one ```json``` block:
{"entities": [{"text": "<a concrete value written in the question: a product, customer, address, staff, supplier, product group, payment method, channel, plate number...>", "role": "<product|customer|address|staff|supplier|group|payment|channel|plate|other>"}],
 "periods": [{"text": "<the time phrase as written>", "kind": "<one kind below>", "<field>": "<value>"}],
 "rules": ["<the exact name (first column) of each RULEBOOK rule the answer depends on>"],
 "answer_type": "<number|percent|string|date|boolean|list|table|unanswerable>",
 "shape": "<one short line: what exactly the answer is, e.g. one number = revenue in VND; product names ordered by quantity, highest first>"}
Period kinds; do not compute dates yourself, the code does (copy a date only when the question writes a full date):
{"kind": "today"} {"kind": "yesterday"} {"kind": "this_week"} {"kind": "last_week"} {"kind": "this_month"} {"kind": "last_month"}
{"kind": "this_quarter"} {"kind": "last_quarter"} {"kind": "ytd"} {"kind": "last_year"}
{"kind": "month", "month": M, "year": Y or null} {"kind": "quarter", "quarter": Q, "year": Y or null} {"kind": "year", "year": Y}
{"kind": "last_n_days", "n": N} {"kind": "date", "date": "YYYY-MM-DD"} {"kind": "range", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} (both days included)
{"kind": "tet", "year": Y} {"kind": "until", "date": "YYYY-MM-DD" or null} (a balance at the end of that day; null = today) {"kind": "other"}
Time phrases are often short or without diacritics: th, thg, t = tháng (T5/2025 is May 2025, a month); q = quý; hqua, bqua = hôm qua; hnay = hôm nay; 6-2025, 06/25, th6/25 = a month.
A phrase relative to today takes its relative kind, never a month number you work out: tháng này, tháng ni, từ đầu tháng đến nay = this_month; tháng trước, tháng liền trước, cùng kỳ tháng trước = last_month; tuần trước = last_week; quý trước = last_quarter; năm ngoái = last_year.
Entities: only concrete values to look up in the data, never generic words (doanh thu, đơn, khách, sản phẩm, sp) and never a time phrase (it goes in periods). Keep the question's spelling. Use [] when there are none."""


def understand_chat(rulebook, ctx, q, anchors):
    return [{"role": "system", "content": UNDERSTAND_SYSTEM + "\n\nRULEBOOK:\n" + rulebook},
            {"role": "user", "content": ctx + "\n\nDATE ANCHORS:\n" + anchors + f"\n\nQUESTION: {q['question']}"}]


def parse_plan(text):
    """The JSON object of an UNDERSTAND reply, {} when there is none."""
    text = text.split("</think>")[-1]
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.S) or [text[text.find("{"):text.rfind("}") + 1]]
    try:
        plan = json.loads(blocks[-1])
    except (json.JSONDecodeError, IndexError):
        return {}
    return plan if isinstance(plan, dict) else {}


def rule_rows(rulebook):
    """norm(rule name) -> 'section > row' for every table row of the rulebook."""
    rows, section = {}, ""
    for line in rulebook.splitlines():
        if line.startswith("#"):
            section = line.lstrip("#").strip()
        elif line.startswith("| ") and not line.startswith(("| Luật", "| Cách nói", "| Type", "| type")):
            name = line.split("|")[1].strip()
            if name and not set(name) <= set("-: "):
                rows[norm(name)] = f"[{section}] {line.strip()}"
    return rows


def value_index(db, meta, max_distinct=20000):
    """Every distinct text value of the shop: (table, column, value, norm(value), its words, row count)."""
    idx = []
    for t in meta["tables"]:
        if t["name"] == "calendar":
            continue
        for col, typ, d, _ in t["columns"]:
            if typ != "VARCHAR" or not d or d > max_distinct:
                continue
            for v, n in db.execute(f"SELECT {quote(col)}, count(*) FROM {quote(t['name'])} WHERE {quote(col)} IS NOT NULL GROUP BY 1").fetchall():
                nv = norm(v)
                if len(nv) >= 2 and not nv.isdigit():
                    idx.append((t["name"], col, str(v), nv, set(nv.split("_")), n))
    return idx


def lookup(idx, text, limit=40):
    """Places matching one entity, best first: (score, rows, table, column, value, how it matches), and how many there are."""
    ne = norm(text)
    if len(ne) < 2:
        return [], 0
    we = set(ne.split("_"))
    out = []
    for t, c, v, nv, wv, n in idx:
        if nv == ne:
            s, how = 1.0, "exactly"
        elif f"_{ne}_" in f"_{nv}_":  # the value contains the phrase as whole words: 'redmi note 14' in 'xiaomi redmi note 14 pro'
            s, how = 0.9, "contains it"
        elif len(wv) >= 2 and f"_{nv}_" in f"_{ne}_":  # the phrase contains the value: 'ca phe sua da size l' has 'ca phe sua da'
            s, how = 0.8, "is part of it"
        else:
            how = "similar spelling"
            if not (we & wv) and not any(a[:3] == b[:3] for a in we if len(a) >= 4 for b in wv if len(b) >= 4):
                continue  # cheap filter before the fuzzy ratio
            s = 0.75 * len(we & wv) / len(we | wv)
            a, b = ne.replace("_", ""), nv.replace("_", "")  # spacing ignored: 'note14' = 'note 14'
            r = max(SequenceMatcher(None, a, b).ratio(), SequenceMatcher(None, a, b[:len(a) + 2]).ratio())  # whole value, or its start
            if r >= 0.8:  # a typo or a missing word: 'xiomi redmi note14' ~ 'xiaomi redmi note 14 128gb'
                s = max(s, 0.85 * r)
            if s < 0.45:
                continue
        out.append((round(s, 2), n, t, c, v, how))
    out.sort(key=lambda x: (-x[0], -x[1]))
    return out[:limit], len(out)


def dated_table(db, meta):
    """(table, date column) of the biggest table with a transaction date; the canonical view when there is one."""
    names = {t["name"] for t in meta["tables"]}
    if "v_sales_all" in names or "v_sales" in names:
        return "v_sales_all", "order_date"
    best = (0, None, None)
    for t in meta["tables"]:
        for col, typ, d, v in t["columns"]:
            if (d is None and typ.startswith(("DATE", "TIMESTAMP")) and v[0] != "None" and re.search(r"ngay|date|thoi_gian|time", ckey(col))
                    and not re.search(r"(^|_)(han|han_dung|het_han|hsd|nsx|sinh|san_xuat|dang_ky|expir\w*)(_|$)", ckey(col))
                    and t["name"] != "calendar" and t["rows"] > best[0]):
                best = (t["rows"], t["name"], col)
    return best[1:]


KINDS = {"today", "yesterday", "this_week", "last_week", "this_month", "last_month", "this_quarter", "last_quarter", "ytd",
         "last_year", "month", "quarter", "year", "last_n_days", "date", "range", "tet", "until", "other"}
EXPLICIT = {"month", "quarter", "year", "date", "range", "tet"}  # named periods: rulebook 7 (outside the data -> unanswerable) applies


def windows(p, as_of, db):
    """Rulebook date windows [start, end) of one period of the plan: [(label, start, end)], start None = from the start
    of the data. A period with a paired window in the rulebook gets both (last month: whole / same days as this month;
    Tết: Tết / the same number of days before it). [] when the kind is unknown or its fields are malformed."""
    d = date.fromisoformat(as_of)
    nxt = d + timedelta(days=1)
    first = lambda y, m: date(y + (m - 1) // 12, (m - 1) % 12 + 1, 1)
    year_of = lambda m: d.year if first(d.year, m) <= d else d.year - 1  # a month/quarter without year: this year's unless after today
    mon, m1 = d - timedelta(days=d.weekday()), d.replace(day=1)
    q1 = first(d.year, 3 * ((d.month - 1) // 3) + 1)
    k = str(p.get("kind", ""))
    try:
        if k == "today":
            return [("", d, nxt)]
        if k == "yesterday":
            return [("", d - timedelta(days=1), d)]
        if k == "this_week":
            return [("Monday to today", mon, nxt)]
        if k == "last_week":
            return [("the whole week before, Monday to Sunday", mon - timedelta(days=7), mon)]
        if k == "this_month":
            return [("day 1 to today", m1, nxt)]
        if k == "last_month":
            pm1 = first(d.year, d.month - 1)
            same = min(pm1 + timedelta(days=d.day), m1)  # rulebook: day 1 to the same day number, or to the end of last month
            return [("whole last month", pm1, m1), ("same days as this month so far, used when compared with this month", pm1, same)]
        if k == "this_quarter":
            return [("quarter start to today", q1, nxt)]
        if k == "last_quarter":
            return [("whole quarter before", first(q1.year, q1.month - 3), q1)]
        if k == "ytd":
            return [("1 January to today", d.replace(month=1, day=1), nxt)]
        if k == "last_year":
            return [("", date(d.year - 1, 1, 1), date(d.year, 1, 1))]
        upto = lambda a, b: [("to today", a, nxt)] if a <= d < b else [("", a, b)]  # the current month/quarter/year runs to today
        if k == "month":
            m = int(p["month"]); y = int(p["year"]) if p.get("year") else year_of(m)
            if first(y, m) == first(d.year, d.month - 1):  # last month written as its number: the rulebook pair still applies
                return windows({"kind": "last_month"}, as_of, db)
            return upto(first(y, m), first(y, m + 1)) if 1 <= m <= 12 else []
        if k == "quarter":
            q = int(p["quarter"]); y = int(p["year"]) if p.get("year") else year_of(3 * q - 2)
            return upto(first(y, 3 * q - 2), first(y, 3 * q + 1)) if 1 <= q <= 4 else []
        if k == "year":
            y = int(p["year"])
            return upto(date(y, 1, 1), date(y + 1, 1, 1))
        if k == "last_n_days":
            n = int(p["n"])
            return [(f"{n} days including today", d - timedelta(days=n - 1), nxt)] if n > 0 else []
        if k == "date":
            x = date.fromisoformat(str(p["date"])[:10])
            return [("", x, x + timedelta(days=1))]
        if k == "range":
            a, b = date.fromisoformat(str(p["start"])[:10]), date.fromisoformat(str(p["end"])[:10])
            return [("both days included", a, b + timedelta(days=1))] if a <= b else []
        if k == "until":
            x = date.fromisoformat(str(p["date"])[:10]) if p.get("date") else d
            return [("the state at the end of that day: everything from the start of the data", None, x + timedelta(days=1))]
        if k == "tet":
            y = int(p.get("year") or d.year)
            a, b = db.execute("SELECT min(date) FILTER (WHERE lunar_year = ? - 1 AND lunar_month = 12 AND lunar_day = 23 AND NOT lunar_leap), "
                              "min(date) FILTER (WHERE lunar_year = ? AND lunar_month = 1 AND lunar_day = 10 AND NOT lunar_leap) FROM calendar", [y, y]).fetchone()
            if not (a and b):
                return []
            n = (b - a).days + 1
            return [(f"Tết {y}, {n} days", a, b + timedelta(days=1)), (f"before Tết (the same {n} days just before)", a - timedelta(days=n), a)]
    except Exception:  # missing / malformed fields, or no calendar table
        return []
    return []


def period_lines(plan, as_of, db):
    """'text = [start, end)' for the named periods of the plan, the form predict.outside() reads."""
    return [f'{p.get("text", "")} = [{a}, {b})' for p in plan.get("periods") or [] if isinstance(p, dict) and p.get("kind") in EXPLICIT
            for _, a, b in windows(p, as_of, db)[:1] if a]


TIME_WORDS = re.compile(r"(?:^|_)(?:hom|ngay|tuan|thang|thg|th|quy|nam|tet|hnay|hqua|bqua|nay|qua|truoc|gan_nhat|gan_day|dau_nam)(?:_|$)|\d_\d{2,4}(?:_|$)")
ENTITY_TIME = re.compile(r"(?:^|_)(?:th|thg|hnay|hqua|bqua|hom_nay|hom_qua|thang_nay|thang_truoc)(?:_|$)|(?:^|_)\d{1,2}_\d{2,4}(?:_|$)")  # time words never part of a name


def validate(plan, question, as_of, db, first=True, idx=None):
    """What is wrong with a reading, as lines for the retry turn; [] when it is usable."""
    if not plan:
        return ["no JSON object was found"]
    probs = []
    for p in plan.get("periods") or []:
        if not isinstance(p, dict):
            probs.append(f"periods must hold objects, got {json.dumps(p, ensure_ascii=False)}")
            continue
        k, t = p.get("kind"), p.get("text", "")
        if k == "other":
            if first:
                probs.append(f'period "{t}" has kind "other": if it is a span of days, write it with a listed kind (a range with start and end '
                             'when nothing else fits); keep "other" only for what is no span of days, e.g. a time of day')
        elif k not in KINDS:
            probs.append(f'period "{t}": kind "{k}" is not one of the listed kinds')
        elif not windows(p, as_of, db):
            probs.append(f'period "{t}": the fields of kind "{k}" are missing or invalid')
        elif first and k in ("date", "range") and not re.search(r"\d[/.-]\d|\d{4}", str(t)):
            probs.append(f'period "{t}": kind "{k}" needs a date written in the question; for a span counted from another period use its '
                         'kind (tet gives the same number of days before Tết, last_n_days counts back from today), else drop it')
        elif k in ("month", "quarter") and re.search(r"\d", str(t)) and str(p.get(k)).lstrip("0") not in re.findall(r"0*(\d+)", str(t)):
            probs.append(f'period "{t}": kind "{k}" with {k} {p.get(k)}, but the text has no {p.get(k)}: check the kind (t, th, thg = tháng; q = quý)')
    words = sorted({m.group().strip("_") for m in TIME_WORDS.finditer(norm(question))})
    if first and not plan.get("periods") and words:
        probs.append(f"periods is empty, but the question has words that may be a time ({', '.join(words)}): list every time phrase it "
                     "names; if it names none, keep periods []")
    if not isinstance(plan.get("entities", []), list):
        probs.append("entities must be a list")
    elif idx is not None:
        for e in plan["entities"]:
            text = e.get("text") if isinstance(e, dict) else e
            if isinstance(text, str) and ENTITY_TIME.search(norm(text)) and all(h[5] != "exactly" for h in lookup(idx, text)[0]):
                probs.append(f'entity "{text}" holds a time phrase, and no value of the data is spelled that way: keep only the value in '
                             'entities and put the time phrase in periods')
    return probs


def fix_message(probs):
    return "PROBLEMS:\n" + "\n".join(f"- {x}" for x in probs) + "\nFix them and write the whole JSON object again in one ```json``` block."


def ground(db, meta, idx, plan, as_of, rules):
    """DATA FACTS text for the SQL prompt: where each value of the plan sits, each period's window and row count, and
    the rulebook rows the plan names. '' when the plan is empty."""
    out = []
    said = [norm(p.get("text", "")) for p in plan.get("periods") or [] if isinstance(p, dict)]
    for e in plan.get("entities") or []:
        text = e.get("text") if isinstance(e, dict) else e
        if not isinstance(text, str) or not text.strip():
            continue
        role = e.get("role") if isinstance(e, dict) else None
        if role in ("period", "time", "date") or any(norm(text) and norm(text) in s for s in said):
            continue  # a time phrase the reading also listed as a value: its window is below
        hits, _ = lookup(idx, text)
        if not hits:
            out.append(f'- "{text}": no value in the shop\'s text columns looks like it (a general word is no filter; a specific name '
                       'that is not there is not in this shop\'s data)')
            continue
        if any(h[5] != "similar spelling" for h in hits):  # the phrase is in the data: near spellings are other items
            hits = [h for h in hits if h[5] != "similar spelling"]
        where = {}  # one line per distinct value, with the columns that hold it
        for _, rows, t, c, v, how in hits:
            where.setdefault((v, how), []).append(f"{quote(t)}.{quote(c)} ({rows} rows)")
        shown = [f"'{v}' ({m}) in " + ", ".join(locs[:3]) for (v, m), locs in list(where.items())[:6]]
        more = len(where) - len(shown)
        out.append(f'- "{text}": ' + "; ".join(shown) + (f"; and {more} more matching values" if more > 0 else ""))
    tab, col = dated_table(db, meta)
    for p in plan.get("periods") or []:
        if not isinstance(p, dict):
            continue
        for label, a, b in windows(p, as_of, db):
            rows = ""
            if tab:
                try:
                    cond = f"{quote(col)} < DATE '{b}'" + (f" AND {quote(col)} >= DATE '{a}'" if a else "")
                    rows = f": {db.execute(f'SELECT count(*) FROM {quote(tab)} WHERE {cond}').fetchone()[0]} rows of {quote(tab)} by {quote(col)}"
                except Exception:
                    rows = ""
            span = f"[{a}, {b})" if a else f"everything before {b}"
            out.append(f'- "{p.get("text", p.get("kind"))}"' + (f" ({label})" if label else "") + f" = {span}{rows}")
    named = []
    for r in plan.get("rules") or []:
        if not isinstance(r, str):
            continue
        key = norm(r)
        row = rules.get(key) or next((v for k, v in rules.items() if k.startswith(key) or SequenceMatcher(None, k, key).ratio() >= 0.85), None)
        if row and row not in named:
            named.append(row)
    text = ""
    if out:
        text += "DATA FACTS (looked up by code in this shop's tables; use these exact values and windows):\n" + "\n".join(out)
    if named:
        text += ("\n\n" if text else "") + "RULES THAT APPLY (from the RULEBOOK):\n" + "\n".join(named)
    return text
