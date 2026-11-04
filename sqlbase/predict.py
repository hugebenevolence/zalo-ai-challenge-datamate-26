"""DataMate inference: K sampled DuckDB queries per question, execute, majority-vote, abstain on low agreement."""
import argparse
import json
import math
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from ingest import ckey, entity_hits, load_business, norm, profile, schema_text
import agent
import debtview
import fewshot
import harness
import llmviews
import prompts

TYPES = ("number", "percent", "string", "date", "boolean", "list", "table", "unanswerable")

SYSTEM = """You are an expert data analyst answering Vietnamese questions about one shop's data with ONE DuckDB SQL query.
The RULEBOOK below defines every business term; follow it exactly (order status, revenue, periods, units, ranking, ties, answer format).
When the rulebook defines a term by a computation (e.g. a shift by the order's time, a main supplier by number of receipts), compute it from the transaction rows that way, even if some table has a column or report with the same name.

How to work:
1. Read the question. In at most 6 short lines, note which rulebook rules apply, which tables/columns hold the data, the exact date window, grain (rows vs distinct orders), status filter and units.
2. Then output exactly:
TYPE: <number|percent|string|date|boolean|list|table|unanswerable>
```sql
<one read-only SELECT/WITH query>
```

SQL rules:
- Use only tables/columns listed in SCHEMA, always double-quoted: "Thành tiền". String literals in single quotes.
- Sales tables are often one row per order line: count orders with COUNT(DISTINCT <order id>); count days, customers, products at their own grain with COUNT(DISTINCT ...). Any event (delivery trip, visit, receipt) is counted the same way: COUNT(DISTINCT <the columns that identify one event>), e.g. COUNT(DISTINCT (order id, trip no)) when the rulebook counts each trip of each order. Quantities: SUM the quantity column, never count rows.
- Order status has four meanings: completed, cancelled (any hủy/huỷ wording, and undeliverable 'không giao được'), returned (trả hàng, đổi trả, hoàn trả, hàng trả về, even partial), in transit (đang giao = a normal order). Spellings vary: classify with replace(strip_accents(lower(col)), 'đ', 'd') LIKE patterns ('%huy%', '%khong giao duoc%', '%tra%'), checking the listed VALUES. A shop without a status column writes it in a note column (Ghi chú); a note not mentioning status = completed.
- Exclude a WHOLE order when it is cancelled or has any returned line if the rule says so (order id NOT IN (...)). Revenue = SUM of line amounts (after line discounts), not the order total after order-level discount.
- Names (products, customers, groups) may differ in case/spaces: compare and group with lower(trim(regexp_replace(x, '\\s+', ' ', 'g'))). Walk-in customers (khách lẻ, kl, lẻ, vãng lai, blank) are not customers: exclude them from customer counts and rankings.
- Use the DATE ANCHORS given; never CURRENT_DATE. Dates: col >= DATE 'a' AND col < DATE 'b' (b exclusive). hôm qua is ONE day. Quarter Q = months 3Q-2..3Q. Filter by the date of the event asked (order date for sales, payment date for payments, import date for imports). Slang: hnay = hôm nay, hqua/bữa qua = hôm qua, dthu = doanh thu, bnhiu/bao nhiu = bao nhiêu.
- Units come from unit columns and conversion tables only, never from numbers inside product names.
- "trả bằng / trả qua / trả tiền mặt / thanh toán bằng" names how an order was PAID (payment method), not a return.
- Status flags (BOOLEAN, when present): "_huy" / "_tra" = this row's status or note is cancelled / returned; "_don_huy" / "_don_tra" = some line of the same order is. Filter status with these flags (e.g. NOT "_don_huy" AND NOT "_don_tra" to drop whole cancelled/returned orders), not by matching status text.
- CANONICAL VIEWS (when listed) apply the rulebook's order rules once: "v_sales" = sold item/service lines of VALID orders only (not cancelled, no returned line) and "v_orders" = VALID orders only, so revenue = SUM(amount) FROM v_sales and order counts = COUNT(*) FROM v_orders need no status filter. Use "v_sales_all" / "v_orders_all" (every line/order, with is_cancelled, is_returned, line_kind) only for questions about cancelled/returned orders, vouchers, shipping or all orders regardless of status. Their columns are listed with them; use the raw tables for anything they lack (imports, payments, debts, stock, units, extra columns).
- number/percent/string/date/boolean: return exactly one row with one column. list: one column, rows in the required order (several numbers asked -> one row is fine). table: first column = object name, then measures in the order asked.
- Answer only what is asked. "top N / N <things> nào / là gì / là ai" asks for names: a list of the names (string when one), no measure columns. Use table only when the question also asks a value for each one (kèm số tiền, mỗi ... bao nhiêu, và số nợ của từng người).
- percent = ROUND(100.0 * part / total, 2). Change "tăng/giảm bao nhiêu %" = ROUND(100.0 * (new - old) / old, 2), negative when it fell. Money = ROUND(x) to whole VND. Other averages ROUND(x, 2).
- A day or month answer is TYPE date (never string): strftime(d, '%Y-%m-%d'), or '%Y-%m' when the question asks which month. boolean: a SQL boolean.
- Names: return the name exactly as stored in the data.
- unanswerable is rare: use it only if NO table/column holds the needed information, or the asked period lies completely outside the date range shown in SCHEMA. A period inside the data range with no matching rows is an answer (0 or empty), not unanswerable. Then write TYPE: unanswerable and SELECT 1."""

# Second wording of the status/revenue/name/date rules. Half the samples use each prompt: voting over both
# beats either alone (dev300 PA@0.5: 0.417 mixed vs 0.34-0.35 single, multi-path self-consistency).
SYSTEM_ALT = re.sub(r"- Order status has.*?(?=\n- Units)", lambda _: """\
- Status/category values vary in spelling and case ('Hủy', 'Đã hủy đơn', 'huỷ'): match every listed variant (e.g. lower(col) IN (...) or strip_accents(lower(col)) LIKE '%huy%'), never just one literal. Names: compare case-insensitively.
- Orders in transit (đang giao) are normal orders. Exclude a WHOLE order when it is cancelled or has any returned line if the rule says so (order id NOT IN (...)).
- Use the DATE ANCHORS given; never CURRENT_DATE. Dates: col >= DATE 'a' AND col < DATE 'b' (b exclusive). Slang: hnay = hôm nay, hqua/bữa qua = hôm qua, dthu = doanh thu, bnhiu/bao nhiu = bao nhiêu.""", SYSTEM, flags=re.S)
SYSTEM_ALT = re.sub(r"- CANONICAL VIEWS.*?\n", "- Compute from the raw shop tables yourself; do not use the v_sales / v_orders views (an independent check of them).\n", SYSTEM_ALT, flags=re.S)
assert SYSTEM_ALT != SYSTEM and "v_orders views (an independent" in SYSTEM_ALT and "CANONICAL VIEWS (when" in SYSTEM


def date_anchors(as_of):
    d = date.fromisoformat(as_of)
    mon = d - timedelta(days=d.weekday())
    m1 = d.replace(day=1)
    pm_end = m1 - timedelta(days=1)
    pm1 = pm_end.replace(day=1)
    pm_same = pm1.replace(day=min(d.day, pm_end.day))
    nxt = lambda x: x + timedelta(days=1)
    return "\n".join([
        f"today (hôm nay) = {d} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d.weekday()]}); half-open end = {nxt(d)}",
        f"the data may go on after today; those later rows have not happened yet (rulebook), so a question naming no period covers the data up to and including today ({d}); a named period keeps its own window; cancel/return status is the order's recorded status",
        f"yesterday (hôm qua, bữa qua, hqua) = {d - timedelta(days=1)} = [{d - timedelta(days=1)}, {d})",
        f"this week (tuần này) = [{mon}, {nxt(d)}); last week (tuần trước) = [{mon - timedelta(days=7)}, {mon})",
        f"this month (tháng này) = [{m1}, {nxt(d)}); last month same days = [{pm1}, {nxt(pm_same)}); whole last month = [{pm1}, {m1})",
        f"year to date (từ đầu năm) = [{d.replace(month=1, day=1)}, {nxt(d)})",
        *(f"quarters of {y}: " + "; ".join(f"Q{i} (quý {i}) = [{y}-{3*i-2:02d}-01, {y + (i == 4)}-{(3*i+1) % 12 or 12:02d}-01)" for i in range(1, 5))
          for y in (d.year - 1, d.year)),
        f"last 7 days = [{d - timedelta(days=6)}, {nxt(d)}); last 30 days = [{d - timedelta(days=29)}, {nxt(d)}); last N days = [today - (N-1) days, {nxt(d)})",
    ])


def period_hints(question, as_of, db=None):
    """Explicit windows for the periods a question names (rulebook: month/quarter without year, Tết by lunar calendar)."""
    d, t, out = date.fromisoformat(as_of), norm(question), []
    first = lambda y, m: date(y + (m - 1) // 12, (m - 1) % 12 + 1, 1)
    def year_of(m_start):  # rulebook: a month/quarter without year is this year's unless it starts after today
        return d.year if first(d.year, m_start) <= d else d.year - 1
    for m, y, m2, y2 in re.findall(r"(?:^|_)(?:(?:thang_|thg_?|th_?)(\d{1,2})(?:_nam|_)?_?(\d{4}|\d{2})?|t(\d{1,2})_(\d{4}|\d{2}))(?=_|$)", t):  # tháng 5, th11/25, T5/2025 (bare T8 is often a product code)
        m, y = int(m or m2), y or y2
        if 1 <= m <= 12:
            y = (int(y) + (2000 if len(y) == 2 else 0)) if y else year_of(m)
            out.append(f"tháng {m}/{y} = [{first(y, m)}, {first(y, m + 1)})")
    # a month written with no word and no day: 03/2026, 1-2026, 8/26 (rulebook "Tháng M/Y"; public: 153 questions had no
    # hint). A 2-digit year must be near today, so 5/12 (a day/month) is left alone; quý 2/2026 is a quarter.
    for mt in re.finditer(r"(?<![\d/.-])(\d{1,2}) ?[/-] ?(20\d\d|\d\d)(?![\d/.-])", question):
        if re.search(r"(?i)(?:^|\W)(?:qu[yý]|q|th[aá]ng|thg|th|t)\s*$", question[:mt.start()]):
            continue
        m, y = mt.groups()
        m, yy = int(m), int(y)
        if 1 <= m <= 12 and (len(y) == 4 or d.year - 2005 <= yy <= d.year - 1999):
            y = yy + (2000 if len(y) == 2 else 0)
            out.append(f"tháng {m}/{y} = [{first(y, m)}, {first(y, m + 1)})")
    for qn, y in re.findall(r"(?:^|_)quy_([1-4])(?:_nam|_)?_?(\d{4})?(?=_|$)", t):
        qn = int(qn); y = int(y) if y else year_of(3 * qn - 2)
        out.append(f"quý {qn}/{y} = [{first(y, 3 * qn - 2)}, {first(y, 3 * qn + 1)})")
    for y in re.findall(r"(?:^|_)nam_(\d{4})(?=_|$)", re.sub(r"(?:thang|quy)_\d{1,2}_nam_\d{4}", "", t)):
        out.append(f"năm {y} = [{y}-01-01, {int(y) + 1}-01-01)")
    if re.search(r"(?:^|_)tet(?=_|$)", t) and db is not None:
        ys = [int(y) for y in re.findall(r"tet(?:_nam)?_(\d{4})", t)] or [d.year]
        for y in ys:
            try:
                a, b = db.execute("SELECT min(date) FILTER (WHERE lunar_year = ? - 1 AND lunar_month = 12 AND lunar_day = 23 AND NOT lunar_leap), "
                                  "min(date) FILTER (WHERE lunar_year = ? AND lunar_month = 1 AND lunar_day = 10 AND NOT lunar_leap) FROM calendar", [y, y]).fetchone()
            except Exception:
                break
            if a and b:
                n = (b - a).days + 1
                out.append(f"Tết {y} = [{a}, {b + timedelta(days=1)}) ({n} days); trước Tết (same {n} days before) = [{a - timedelta(days=n)}, {a})")
    for n in re.findall(r"(?:^|_)(\d{1,3})_ngay_(?:gan_nhat|gan_day|qua|vua_qua|truoc|cuoi)", t):
        n = int(n)
        out.append(f"{n} ngày gần nhất (incl. today) = [{d - timedelta(days=n - 1)}, {d + timedelta(days=1)})")
    for dd, mm, yy in re.findall(r"(?:^|_)(\d{1,2})_(\d{1,2})_(\d{4})(?=_|$)", t):
        try:
            x = date(int(yy), int(mm), int(dd)); out.append(f"ngày {x} = [{x}, {x + timedelta(days=1)})")
        except ValueError:
            pass
    return list(dict.fromkeys(out))


def data_span(meta):
    """Date span of the shop's biggest table that has a transaction-date column (not expiry/birth dates)."""
    best = (0, None, None)
    for t in meta["tables"]:
        ds = [v for col, typ, d, v in t["columns"] if d is None and typ.startswith(("DATE", "TIMESTAMP")) and v[0] != "None"
              and re.search(r"ngay|date|thoi_gian|time", ckey(col)) and not re.search(r"(^|_)(han|han_dung|het_han|hsd|nsx|sinh|san_xuat|expir\w*)(_|$)", ckey(col))]
        if ds and t["name"] != "calendar" and t["rows"] > best[0]:
            best = (t["rows"], min(v[0][:10] for v in ds), max(v[1][:10] for v in ds))
    return best[1:]


def outside(periods, span):
    """Rulebook 7: every named period lies completely outside the shop's data dates -> unanswerable, not 0.
    Train: fires on 14/3000 questions, all 14 gold unanswerable."""
    w = [m for p in periods for m in re.findall(r"\[(\d{4}-\d\d-\d\d), (\d{4}-\d\d-\d\d)\)", p)[:1]]
    lo, hi = span
    return bool(w and lo) and all(e <= lo or s > hi for s, e in w)


# Vietnamese chat abbreviations (language-level, not shop-specific), glossed next to the question when present
SLANG = {"hqua": "hôm qua", "hnay": "hôm nay", "hnao": "hôm nào", "bqua": "bữa qua", "dthu": "doanh thu",
         "bnhiu": "bao nhiêu", "bn": "bao nhiêu", "bnh": "bao nhiêu", "sl": "số lượng", "sp": "sản phẩm", "kh": "khách hàng",
         "ko": "không", "k": "không", "dc": "được", "đc": "được", "tg": "thời gian", "nv": "nhân viên", "ncc": "nhà cung cấp",
         "th": "tháng", "thg": "tháng", "z": "vậy", "mik": "mình", "mk": "mình", "j": "gì",
         "gồi": "rồi", "rùi": "rồi", "hok": "không", "hem": "không", "nhìu": "nhiều", "bít": "biết", "lun": "luôn", "jup": "giúp", "dzậy": "vậy", "zậy": "vậy"}


def chat(system, biz_ctx, q, shots=""):
    hits = q["_hits"]
    words = re.findall(r"\w+", q["question"].lower())
    gloss = [f"{w} = {SLANG[w]}" for w in dict.fromkeys(words) if w in SLANG]
    user = (biz_ctx + "\n\nDATE ANCHORS:\n" + date_anchors(q["as_of_date"])
            + "".join("\n" + p for p in q.get("_periods", []))
            + ("\n\nVALUES MATCHING THE QUESTION:\n" + "\n".join(hits) if hits else "")
            + ("\n\n" + q["_facts"] if q.get("_facts") else "")
            + ("\n\n" + shots if shots else "")
            + f"\n\nQUESTION: {q['question']}" + (f"\n(abbreviations: {'; '.join(gloss)})" if gloss else ""))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse(text):
    if "<think>" in text:
        if "</think>" not in text:
            return None, None
        text = text.split("</think>")[-1]
    t = re.search(r"TYPE:\s*([a-z]+)", text)
    s = re.findall(r"```(?:sql)?\s*(.*?)```", text, re.S)
    return (t.group(1) if t else None), (s[-1].strip() if s else None)


def cell(v):
    if isinstance(v, (list, tuple)):
        return [cell(x) for x in v]
    if isinstance(v, dict):
        return {str(k): cell(x) for k, x in v.items()}
    if isinstance(v, Decimal):
        v = float(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError("non-finite number")
        return int(v) if v.is_integer() else round(v, 4)
    if isinstance(v, (datetime, date)):
        return v.isoformat()[:10]
    return v


def run_sql(db, sql, timeout=20):
    timer = threading.Timer(timeout, db.interrupt)
    timer.start()
    try:
        cur = db.execute(sql)
        names = [c[0] for c in cur.description]
        rows = cur.fetchmany(2001)
    finally:
        timer.cancel()
    if len(rows) > 2000:
        raise ValueError("result too large")
    return names, [[cell(v) for v in r] for r in rows]


def to_answer(kind, names, rows):
    if kind == "table":
        return {"columns": names, "rows": rows}
    if kind == "list":
        if len(names) == 1:
            out = [r[0] for r in rows]
        elif len(rows) == 1:
            out = rows[0]
        else:
            raise ValueError(f"list needs one column, got {names}")
        if any(x is None or isinstance(x, (bool, list, dict)) for x in out):  # the format check rejects such a list
            raise ValueError("every list item must be one non-NULL name or number (one row per item, one column)")
        return out
    if len(rows) != 1 or len(names) != 1:
        raise ValueError(f"expected one cell, got {len(rows)} rows x {names}")
    v = rows[0][0]
    if v is None:
        raise ValueError("result is NULL (check filters/values)")
    if kind in ("number", "percent"):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("expected a number")
        # rulebook: money (incl. money averages) rounds to whole đồng, half up; other averages to 2 decimals.
        # shortcut: |v| >= 1000 taken as VND money; a non-money average that large would be wrongly rounded.
        v = float(Decimal(str(v)).quantize(Decimal("1" if kind == "number" and abs(v) >= 1000 else ".01"), rounding=ROUND_HALF_UP))
        return int(v) if v.is_integer() else v
    if kind == "boolean":
        if not isinstance(v, bool):
            raise ValueError("expected boolean")
        return v
    v = str(v)
    if kind == "date" and not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", v):
        raise ValueError("date must be YYYY-MM-DD or YYYY-MM")
    return v


CHANGE = re.compile(r"(?:^|_)(?:tang|giam|thay_doi|hon_hay_kem|cao_hay_thap|len_xuong|tang_truong|chenh_lech)(?:_|$)")


def execute(db, text, question=""):
    kind, sql = parse(text)
    if kind not in TYPES:
        raise ValueError("missing or invalid TYPE line")
    if kind == "unanswerable":
        return {"type": kind}
    if not sql or not re.match(r"(?is)\s*(with|select|from)\b", sql) or re.search(r"(?i)\b(insert|update|delete|create|drop|alter|copy|attach|install|load|pragma)\b", sql):
        raise ValueError("need one read-only SELECT query in ```sql```")
    sql = re.sub(r'"(%[^"]*)"', r"'\1'", sql)  # a double-quoted strftime format is a string, never a column
    if kind == "percent" and CHANGE.search(norm(question)) and "-" not in re.sub(r"'[^']*'", "", sql):
        # 14B often writes 100*new/old after planning 100*(new-old)/old; lint it into the repair turn
        raise ValueError("a change in % is ROUND(100.0 * (new - old) / old, 2), negative when it fell; this query has no subtraction")
    names, rows = run_sql(db, sql.rstrip(";"))
    ans = to_answer(kind, names, rows)
    if kind == "string" and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", ans):  # a day/month answer is type date
        kind = "date"
    return {"type": kind, "answer": ans}


def repair_hint(err):
    """One generic line for the commonest failure kinds (public trace: 1,933 of 9,912 samples still failed after one repair,
    mostly an empty result or a result of the wrong shape)."""
    if err.startswith("expected one cell, got 1 rows"):
        return "Return only the one value asked (a single column, e.g. just the name), or use TYPE list/table if the question asks for several values.\n"
    if err.startswith(("expected one cell, got 0 rows", "result is NULL")):
        return ("No row matched the filters: check every filter value against the SCHEMA values and VALUES MATCHING THE QUESTION (spelling, "
                "or the value sits in another column or table, e.g. an address in the customer list), and the date window against DATE ANCHORS.\n")
    if err.startswith("expected one cell, got"):
        return "The question asks for one value: aggregate, or ORDER BY ... LIMIT 1 for a top-1 question; use TYPE list/table if it asks for several.\n"
    if "not found in FROM clause" in err:
        return "Use only the columns SCHEMA lists for each table, double-quoted; a column of another table needs a JOIN on its key.\n"
    return ""


def vote(preds, k):
    """Majority over executed answers; confidence = share of all K samples agreeing. A number and a percent of the same
    value are one answer: Qwen3.5 labels a % change either way (public: 38 abstains were such a split), so the group
    takes its commoner label, percent on a tie, and that label's commonest value."""
    dump = lambda p: json.dumps(p, ensure_ascii=False, default=str, sort_keys=True)
    groups = defaultdict(list)
    for p in preds:
        if p:
            a = p.get("answer")
            num = p["type"] in ("number", "percent") and isinstance(a, (int, float)) and not isinstance(a, bool)
            groups[dump(["num", round(float(a), 2)]) if num else dump(p)].append(p)
    if not groups:
        return {"type": "abstain"}, 0.0
    best = max(groups.values(), key=len)  # first seen on a tie, as Counter.most_common
    types = Counter(p["type"] for p in best)
    kind = "percent" if types["percent"] and types["percent"] >= types["number"] else max(types, key=types.get)
    pick = Counter(dump(p) for p in best if p["type"] == kind).most_common(1)[0][0]
    return json.loads(pick), len(best) / k


def decide(ans, conf, threshold):
    """Answer only with enough agreement. Abstain on unanswerable votes (0/11 correct, dev300) and on 0/empty
    number/percent/list/table answers: train gold has 4 such in 2458, the vote had 0/63 correct across traces."""
    nil = ans.get("answer") in (0, [], None) or ans["type"] == "table" and not ans["answer"]["rows"]
    return conf >= threshold and ans["type"] not in ("unanswerable", "abstain") and not (nil and ans["type"] != "boolean")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--k", type=int, default=16, help="samples per question, split over the two prompts")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--threshold", type=float, default=0.5, help="min vote share to answer")
    p.add_argument("--ids", type=Path, help="optional JSONL/TXT of question ids to run")
    p.add_argument("--limit", type=int)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-mem", type=float, default=0.9)
    p.add_argument("--quant")
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--views", choices=["llm", "none"], default="llm", help="canonical sales tables written by the LLM per shop (llmviews.py), or none")
    p.add_argument("--view-think", type=int, default=6144, help="thinking-mode token budget for the per-shop view stage (0 = no thinking)")
    p.add_argument("--views-from", type=Path, help="reuse the view SELECTs an earlier run chose (its logs/_views_<shop>.json) instead of drafting")
    p.add_argument("--repairs", type=int, default=2, help="greedy repair turns for a failed sample (thinking samples included)")
    p.add_argument("--lora-thinking", action="store_true", help="also use --lora for the thinking samples")
    p.add_argument("--lora", type=Path, help="LoRA adapter for the non-thinking SQL samples, TYPE fills and every repair turn (view drafts and thinking samples stay on the base model)")
    p.add_argument("--prompt", choices=["v2", "v3"], default="v2", help="v3: the locked prompt of prompts.py on every sample")
    p.add_argument("--observe", action="store_true", help="after the samples ran, one greedy turn where each sample sees its result and keeps or corrects it (harness.py)")
    p.add_argument("--think-retry", action="store_true", help="a thinking sample that hit the budget is sampled again without thinking")
    p.add_argument("--fewshot", type=Path, help="example bank (runs/analysis/build_fewshot_bank.py): the second half of the samples get similar train examples instead of the raw-tables prompt")
    p.add_argument("--fewshot-k", type=int, default=4)
    p.add_argument("--debt-views", action="store_true", help="canonical debt tables + debt_at(day) written by the LLM per shop (debtview.py)")
    p.add_argument("--save-prompts", action="store_true", help="keep each question's user message in the trace (SFT data from train runs)")
    p.add_argument("--agent", action="store_true", help="UNDERSTAND + GROUND before the SQL samples (agent.py): looked-up values, date windows, rule rows")
    p.add_argument("--shard", default="0/1", help="i/n: run the i-th of n business shards (one process per GPU)")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--think-k", type=int, default=0, help="extra thinking-mode samples for questions whose vote share is below --think-below")
    p.add_argument("--think-below", type=float, default=0.75)
    p.add_argument("--think-budget", type=int, default=4096)
    p.add_argument("--deadline", type=float, default=0, help="seconds for this process; thinking is skipped for a shop once the run is behind a pro-rata pace")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--engine", type=json.loads, default={}, help='extra vLLM engine kwargs as JSON, e.g. {"kv_cache_dtype": "fp8"}')
    a = p.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    t0 = time.time()
    qs = [json.loads(l) for l in (a.data / "questions.jsonl").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    if a.ids:
        keep = set(re.findall(r"q_[0-9a-f]+", a.ids.read_text()))
        qs = [q for q in qs if q["id"] in keep]
    qs = qs[:a.limit] if a.limit else qs
    a.output.mkdir(parents=True, exist_ok=True)
    rulebook = (a.data / "rulebook.md").read_text(encoding="utf-8-sig")
    base_prompts = (prompts.SYSTEM_V3, prompts.SYSTEM_V3) if a.prompt == "v3" else (SYSTEM, SYSTEM) if a.fewshot else (SYSTEM, SYSTEM_ALT)
    systems = [s + "\n\nRULEBOOK:\n" + rulebook for s in base_prompts]
    bank = fewshot.Bank(a.fewshot) if a.fewshot else None
    rules = agent.rule_rows(rulebook)
    half = a.k // 2

    shops = sorted({q["business_id"] for q in qs})
    i, n = map(int, a.shard.split("/"))
    load = Counter(q["business_id"] for q in qs)
    bins = [[] for _ in range(n)]
    for b in sorted(shops, key=lambda b: (-load[b], b)):  # greedy balance of question counts over shards
        min(bins, key=lambda x: sum(load[y] for y in x)).append(b)
    shops = sorted(bins[i])
    qs = [q for q in qs if q["business_id"] in shops]
    write_submission(a.output / "submission.jsonl", qs, {})  # valid all-abstain file in case the run is killed
    by_shop = defaultdict(list)
    for q in qs:
        by_shop[q["business_id"]].append(q)
    (a.output / "logs").mkdir(exist_ok=True)

    def load_all():  # runs in a thread while vLLM starts (the engine is a separate process)
        loaded = {}
        for b in shops:
            try:
                db, meta = load_business(a.data, b)
                db.execute("SET enable_external_access = false")  # model SQL cannot read files or the network
                db.execute("SET memory_limit = '6GB'")  # a runaway model join fails (-> repair) instead of exhausting host RAM
                loaded[b] = db, meta
            except Exception as e:
                print(f"[load error] {b}: {e}", flush=True)
        return loaded
    loading = ThreadPoolExecutor(1).submit(load_all)

    llm = LLM(model=a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem, max_model_len=a.context,
              enable_prefix_caching=True, disable_log_stats=False, seed=a.seed, language_model_only=True, **({"quantization": a.quant} if a.quant else {}),
              **({"enable_lora": True, "max_lora_rank": 64, "max_loras": 1} if a.lora else {}), **a.engine)
    tok = llm.get_tokenizer()
    kw = {"chat_template_kwargs": {"enable_thinking": False}, "use_tqdm": False}
    lkw = dict(kw, lora_request=LoRARequest("sql", 1, str(a.lora))) if a.lora else kw  # the SQL-sample calls
    print(f"model ready in {time.time() - t0:.0f}s; shard {a.shard}: {len(shops)} shops, {len(qs)} questions", flush=True)
    loaded = loading.result()
    print(f"loaded {len(loaded)} shops in {time.time() - t0:.0f}s", flush=True)
    vtexts, vfixed = {}, {}
    if a.views == "llm" and a.views_from:  # same views as that run, so an A/B run differs only in what it changes
        vfixed = {b: json.loads(f.read_text(encoding="utf-8")) for b in loaded if (f := a.views_from / "logs" / f"_views_{b}.json").exists()}
        print(f"views of {len(vfixed)}/{len(loaded)} shops reused from {a.views_from}", flush=True)
    drafted = [b for b in loaded if b not in vfixed]  # shops without reused views get fresh drafts
    if a.views == "llm" and drafted:  # per shop: a.k drafts of one mapping SELECT, all shops in one batch (GPU stays full)
        vchats = [[{"role": "system", "content": llmviews.VIEW_SYSTEM + "\n\nRULEBOOK:\n" + rulebook},
                   {"role": "user", "content": "README:\n" + loaded[b][1]["readme"] + "\n\nSCHEMA:\n" + schema_text(loaded[b][1])}] for b in drafted]
        vsp = SamplingParams(n=a.k, temperature=0.6 if a.view_think else a.temperature, top_p=0.95, top_k=20,
                             max_tokens=a.view_think + 2048, seed=a.seed)
        vouts = llm.chat(vchats, vsp, chat_template_kwargs={"enable_thinking": bool(a.view_think)}, use_tqdm=False)
        vtexts = {b: [x.text for x in o.outputs] for b, o in zip(drafted, vouts)}
        vchecked = {b: llmviews.check(loaded[b][0], ts) for b, ts in vtexts.items()}
        # drafts failing the data tests get one repair with the error, all shops in one batch
        fail = [(b, j) for b, c in vchecked.items() for j, (fp, _) in enumerate(c) if fp is None]
        if fail:
            vchat_of = dict(zip(drafted, vchats))
            rch = [vchat_of[b] + [{"role": "assistant", "content": vtexts[b][j].split("</think>")[-1].strip()},
                                  {"role": "user", "content": f"ERROR: {vchecked[b][j][1]}\nFix the SELECT. Answer with the SQL only, in one ```sql``` block."}]
                   for b, j in fail]
            routs = llm.chat(rch, SamplingParams(n=1, temperature=0, max_tokens=2048), **kw)
            for (b, j), o in zip(fail, routs):
                vtexts[b][j] += "\n#REPAIR\n" + o.outputs[0].text
                vchecked[b][j] = llmviews.check(loaded[b][0], [o.outputs[0].text])[0]
        print(f"view drafts in {time.time() - t0:.0f}s ({len(fail)} repaired)", flush=True)
    dchecked, dfixed = {}, {}
    if a.debt_views:  # same recipe as the views: a.k drafts per shop, data tests, one repair turn, most common fingerprint
        if a.views_from:
            dfixed = {b: json.loads(f.read_text(encoding="utf-8")) for b in loaded if (f := a.views_from / "logs" / f"_debt_{b}.json").exists()}
        ddraft = [b for b in loaded if b not in dfixed]
        if ddraft:
            dchats = {b: [{"role": "system", "content": debtview.DEBT_SYSTEM + "\n\nRULEBOOK:\n" + rulebook},
                          {"role": "user", "content": "README:\n" + loaded[b][1]["readme"] + "\n\nSCHEMA:\n" + schema_text(loaded[b][1])}] for b in ddraft}
            douts = llm.chat(list(dchats.values()), SamplingParams(n=a.k, temperature=a.temperature, top_p=0.95, top_k=20, max_tokens=2048, seed=a.seed), **kw)
            dtexts = {b: [x.text for x in o.outputs] for b, o in zip(ddraft, douts)}
            dchecked = {b: debtview.check(loaded[b][0], ts) for b, ts in dtexts.items()}
            dfail = [(b, j) for b, c in dchecked.items() for j, (fp, _) in enumerate(c) if fp is None]
            if dfail:
                rch = [dchats[b] + [{"role": "assistant", "content": dtexts[b][j].strip()},
                                    {"role": "user", "content": f"ERROR: {dchecked[b][j][1]}\nFix it. Answer with the two ```sql``` blocks only, or NONE."}] for b, j in dfail]
                for (b, j), o in zip(dfail, llm.chat(rch, SamplingParams(n=1, temperature=0, max_tokens=2048), **kw)):
                    dtexts[b][j] += "\n#REPAIR\n" + o.outputs[0].text
                    dchecked[b][j] = debtview.check(loaded[b][0], [o.outputs[0].text])[0]
            print(f"debt drafts in {time.time() - t0:.0f}s ({len(dfail)} repaired)", flush=True)
    results, n_repair = {}, 0
    tr = (a.output / "trace.jsonl").open("w", encoding="utf-8")
    for b, (db, meta) in loaded.items():  # one shop at a time: answer all its questions, write results (checkpoint), close
        tb = time.time()
        view, vinfo = "", {}
        if b in vfixed:
            sql, nv, verr = vfixed[b]["sql"], vfixed[b]["agree"], vfixed[b]["errors"]
            vtexts[b] = vfixed[b]["drafts"]
            if sql:
                llmviews.build(db, sql)
        elif b in vtexts:
            sql, nv = llmviews.pick(db, vchecked[b])
            verr = [e for fp, e in vchecked[b] if fp is None]
        if b in vtexts:
            if sql and nv * 4 >= a.k:  # too little agreement = the mapping is a guess; better no view than a wrong one
                view = llmviews.describe(db, sql, nv, a.k)
                meta["tables"] += [profile(db, v, "canonical view") for v in ("v_sales", "v_orders")]
            vinfo = {"agree": nv, "k": a.k, "used": bool(view), "sql": sql, "errors": verr, "drafts": vtexts[b]}
            print(f"[views] {b}: {nv}/{a.k} agree, used={bool(view)}, {len(verr)} failed drafts", flush=True)
        debt = ""
        if a.debt_views:
            sqls, dn = (dfixed[b]["sqls"], dfixed[b]["agree"]) if b in dfixed else debtview.pick(dchecked[b])
            dfps = dfixed[b].get("checked", []) if b in dfixed else [str(fp) for fp, _ in dchecked.get(b, [])]
            dk = dfixed[b].get("k", a.k) if b in dfixed else a.k  # a reused file keeps its own verdict and draft count: re-judging it at another k, or from a reuse that lost "checked", dropped a working layer
            if sqls and (dfixed[b]["used"] if b in dfixed else debtview.accept(dfps, dn, dk)):
                debtview.build(db, sqls)  # the tables stay out of SCHEMA: the prompt offers the macros and their recipes
                debt = debtview.describe(dn, dk)
            (a.output / "logs" / f"_debt_{b}.json").write_text(json.dumps({"agree": dn, "k": dk, "used": bool(debt), "sqls": sqls, "checked": dfps,
                                                                         "errors": [e for fp, e in dchecked.get(b, []) if fp is None]}, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[debts] {b}: {dn}/{dk} agree, used={bool(debt)}", flush=True)
        ctx = "README:\n" + meta["readme"] + "\n\nSCHEMA:\n" + schema_text(meta) + ("\n\nCANONICAL VIEWS:\n" + view if view else "") + ("\n\n" + debt if debt else "")
        todo = by_shop[b]
        for q in todo:
            q["_hits"] = entity_hits(db, meta, q["question"])
            q["_periods"] = period_hints(q["question"], q["as_of_date"], db)
            q["_oor"] = outside(q["_periods"], data_span(meta))
        if a.agent:  # UNDERSTAND: a greedy reading per question, re-asked with its problems (unknown period kind, bad fields,
            # no JSON) up to --repairs times; GROUND: code looks its values up and turns its periods into rulebook windows.
            # The windows replace the regex period hints, so no time phrase needs a pattern written for it.
            idx = agent.value_index(db, meta)
            convs = {i: agent.understand_chat(rulebook, ctx, q, date_anchors(q["as_of_date"])) for i, q in enumerate(todo)}
            ask = [i for i, c in convs.items()
                   if len(tok.apply_chat_template(c, tokenize=True, add_generation_prompt=True, return_dict=False)) <= a.context - 832]
            plans, turns = {}, {}
            for turn in range(a.repairs + 1):
                if not ask:
                    break
                outs = llm.chat([convs[i] for i in ask], SamplingParams(n=1, temperature=0, max_tokens=768), **kw)
                again = []
                for i, o in zip(ask, outs):
                    reply = o.outputs[0].text
                    plans[i], turns[i] = agent.parse_plan(reply), turn + 1
                    probs = agent.validate(plans[i], todo[i]["question"], todo[i]["as_of_date"], db, first=turn == 0, idx=idx)
                    if probs and turn < a.repairs:
                        convs[i] = convs[i] + [{"role": "assistant", "content": reply.split("</think>")[-1].strip()},
                                               {"role": "user", "content": agent.fix_message(probs)}]
                        again.append(i)
                ask = again
            for i, q in enumerate(todo):
                q["_plan"] = dict(plans.get(i, {}), _turns=turns.get(i, 0))
                q["_facts"] = agent.ground(db, meta, idx, q["_plan"], q["as_of_date"], rules)
                q["_periods"] = []  # the windows are in the facts now
                q["_oor"] = outside(agent.period_lines(q["_plan"], q["as_of_date"], db), data_span(meta))
        # examples read the canonical views, so a shop without them gets none
        chats = [[chat(s, ctx, q, fewshot.block(bank.retrieve(q["question"], b, a.fewshot_k, debt=bool(debt))) if bank and view and j == 1 else "") for j, s in enumerate(systems)] for q in todo]
        budget = a.context - a.max_tokens - 64
        fits = [max(len(tok.apply_chat_template(c, tokenize=True, add_generation_prompt=True, return_dict=False)) for c in cs) <= budget
                for cs in chats]
        tg = time.time()
        sp = SamplingParams(n=half, temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens, seed=a.seed)
        outs = llm.chat([c for cs, f in zip(chats, fits) if f for c in cs], sp, **lkw)
        it = iter(outs)
        texts = [[o.text for _ in systems for o in next(it).outputs] if f else [] for f in fits]

        # Execute; failed samples get greedy repair turns with the error message (fix below).
        preds = [[None] * len(t) for t in texts]
        errs, repair = [[None] * len(t) for t in texts], []
        for i, ts in enumerate(texts):
            for j, t in enumerate(ts):
                try:
                    preds[i][j] = execute(db, t, todo[i]["question"])
                except Exception as e:
                    errs[i][j] = str(e)[:400]
                    repair.append((i, j))
        # A sample with SQL but no TYPE line (14B: ~30% of samples) gets the model's own type: prefill "TYPE:" and read <=4 tokens.
        notype = [(i, j) for i, j in repair if errs[i][j].startswith("missing or invalid TYPE") and parse(texts[i][j])[1]]
        if notype:
            tchats = [chats[i][j // half] + [{"role": "assistant", "content": texts[i][j].rstrip() + "\nTYPE:"}] for i, j in notype]
            touts = llm.chat(tchats, SamplingParams(n=1, temperature=0, max_tokens=4), add_generation_prompt=False,
                             continue_final_message=True, **lkw)
            for (i, j), o in zip(notype, touts):
                texts[i][j] = texts[i][j].rstrip() + "\nTYPE:" + o.outputs[0].text
                try:
                    preds[i][j], errs[i][j] = execute(db, texts[i][j], todo[i]["question"]), None
                    repair.remove((i, j))
                except Exception as e:
                    errs[i][j] = str(e)[:400]
        def fix(pairs, prompt_of):
            """Greedy repair turns for failed samples (i, j): the error plus a hint for its kind, up to --repairs turns."""
            conv = {(i, j): chats[i][prompt_of(j)] + [{"role": "assistant", "content": texts[i][j].split("</think>")[-1].strip()}]
                    for i, j in pairs}
            turns = 0
            for _ in range(a.repairs):
                pairs = [(i, j) for i, j in pairs if preds[i][j] is None]
                if not pairs:
                    break
                for i, j in pairs:
                    e = errs[i][j].split(" | ")[-1]
                    conv[i, j].append({"role": "user", "content": f"ERROR: {e}\n{repair_hint(e)}Fix the query. Answer again in the same format."})
                routs = llm.chat([conv[ij] for ij in pairs], SamplingParams(n=1, temperature=0, max_tokens=a.max_tokens), **lkw)
                for (i, j), o in zip(pairs, routs):
                    out = o.outputs[0].text
                    texts[i][j] += "\n#REPAIR\n" + out
                    conv[i, j].append({"role": "assistant", "content": out})
                    try:
                        preds[i][j] = execute(db, out, todo[i]["question"])
                    except Exception as e:
                        errs[i][j] += " | " + str(e)[:400]
                turns += len(pairs)
            return turns

        prompt_idx = lambda j: j // half if j < 2 * half else (1 if bank else (j - 2 * half) % len(systems))  # which prompt asked sample j
        n_repair += fix(repair, lambda j: j // half)
        behind = a.deadline and time.time() - t0 > a.deadline * (len(results) + len(todo)) / max(1, len(qs))
        if a.think_k and not behind:  # adaptive compute: low-agreement questions get thinking-mode samples (alternating prompts)
            hard = [i for i, ps in enumerate(preds) if texts[i] and vote(ps, len(ps))[1] < a.think_below]
            if hard:
                # with examples available all thinking samples get them: right 86% vs 71.5% on 873 hard train questions
                hchats = [chats[i][1 if bank else s % len(systems)] for i in hard for s in range(a.think_k)]
                houts = llm.chat(hchats, SamplingParams(n=1, temperature=0.6, top_p=0.95, top_k=20, max_tokens=a.think_budget + a.max_tokens, seed=a.seed),
                                 chat_template_kwargs={"enable_thinking": True}, use_tqdm=False, **({"lora_request": lkw["lora_request"]} if a.lora and a.lora_thinking else {}))
                for n, o in enumerate(houts):
                    i, t = hard[n // a.think_k], o.outputs[0].text
                    if "<think>" not in t:  # Qwen3.5's template opens <think> in the prompt; parse() keys on it to spot a cut-off thought
                        t = "<think>\n" + t
                    texts[i].append(t)
                    try:
                        preds[i].append(execute(db, t, todo[i]["question"])); errs[i].append(None)
                    except Exception as e:
                        preds[i].append(None); errs[i].append(str(e)[:400])
                if a.think_retry:  # a thought that hit the budget has no query: ask again without thinking (5% of thinking samples, half of them loops)
                    cut = [(i, j) for i in hard for j in range(2 * half, len(texts[i])) if "</think>" not in texts[i][j]]
                    if cut:
                        rt = llm.chat([chats[i][prompt_idx(j)] for i, j in cut], SamplingParams(n=1, temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens, seed=a.seed), **lkw)
                        for (i, j), o in zip(cut, rt):
                            texts[i][j] = o.outputs[0].text
                            try:
                                preds[i][j], errs[i][j] = execute(db, texts[i][j], todo[i]["question"]), None
                            except Exception as e:
                                preds[i][j], errs[i][j] = None, str(e)[:400]
                # failed thinking samples get the same repair turns (non-thinking); a cut-off thought has no query to fix
                n_repair += fix([(i, j) for i in hard for j in range(2 * half, len(texts[i])) if preds[i][j] is None and ("</think>" in texts[i][j] or a.think_retry)], prompt_idx)
        observed = {}
        if a.observe:
            known = harness.known_values(db)
            items = {(i, j): (chats[i][prompt_idx(j)] + [{"role": "assistant", "content": t.split("\n#REPAIR\n")[-1].split("</think>")[-1].strip()}], t, preds[i][j], todo[i]["question"])
                     for i, ts in enumerate(texts) for j, t in enumerate(ts) if preds[i][j]}
            gen = lambda cs: [o.outputs[0].text for o in llm.chat(cs, SamplingParams(n=1, temperature=0, max_tokens=a.max_tokens), **lkw)]
            observed, revised = harness.observe_turn(gen, execute, run_sql, parse, db, known, items)
            for (i, j), (t, new) in revised.items():
                texts[i][j], preds[i][j] = t, new
            print(f"[observe] {b}: {len(observed)} samples, {len(revised)} revised", flush=True)
        per_q = (time.time() - tb) / max(1, len(todo))  # batched generation: shop wall time shared equally (view batch excluded)
        for qi, (q, ts, ps, es, cs) in enumerate(zip(todo, texts, preds, errs, chats)):
            ans, conf = vote(ps, len(ps))
            r = ({"type": "unanswerable", "reason": "period outside the shop's data dates"} if q["_oor"]
                 else ans if decide(ans, conf, a.threshold) else {"type": "abstain"})
            results[q["id"]] = r
            rec = {"id": q["id"], "business_id": b, "question": q["question"], "final": r, "vote": ans, "conf": conf,
                   "elapsed_seconds": round(per_q, 2), "samples": [dict({"text": t, "pred": pr, "error": e}, **({"observed": observed[qi, j]} if (qi, j) in observed else {})) for j, (t, pr, e) in enumerate(zip(ts, ps, es))]}
            if a.agent:
                rec.update(plan=q.get("_plan"), facts=q.get("_facts"))
            if a.save_prompts:  # the system message is SYSTEM / SYSTEM_ALT + rulebook (samples 0..K/2-1 / the rest), the user message is shared
                rec["prompt_user"] = cs[0][1]["content"]
            tr.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            (a.output / "logs" / f"{q['id']}.json").write_text(json.dumps(rec, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
        tr.flush()
        (a.output / "logs" / f"_views_{b}.json").write_text(json.dumps(vinfo, ensure_ascii=False, indent=1), encoding="utf-8")
        write_submission(a.output / "submission.jsonl", qs, results)
        db.close()
        print(f"[{b}] {len(todo)} q, {sum(r['type'] != 'abstain' for r in (results[q['id']] for q in todo))} answered, "
              f"{len(repair)} repairs, gen {time.time() - tg:.0f}s, total {time.time() - tb:.0f}s{', behind pace: no thinking' if behind else ''}", flush=True)
    tr.close()
    run = {"args": {k: str(v) for k, v in vars(a).items()}, "questions": len(qs), "seconds": round(time.time() - t0),
           "answered": sum(r["type"] != "abstain" for r in results.values()), "repairs": n_repair}
    (a.output / "run.json").write_text(json.dumps(run, indent=2))
    print(run, flush=True)


def write_submission(path, qs, results):
    """Atomic rewrite: every question id, abstain when not answered yet."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps({"id": q["id"], **results.get(q["id"], {"type": "abstain"})}, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    main()
