"""Per-shop debt layer written by the LLM, same method as llmviews: the model reads one shop's README + SCHEMA + RULEBOOK and
maps its raw tables to two fixed tables (orders that can carry a debt, payments). Generic SQL below derives the debt of every
order at any day. All drafts are executed; the most common result fingerprint wins. A rulebook debt definition on this layer
matched 8/8 gold answers on biz_123, where the free SQL samples got 1/14 debt questions right on holdout150."""
import re
import threading

DEBT_SYSTEM = """You map ONE shop's raw tables (DuckDB) to two canonical debt tables. Read README, SCHEMA and the RULEBOOK section on debts: they say which table holds the orders, what a customer owes on an order, and where payments are recorded.
If the shop records no customer debts and no payments at all, answer with the single word NONE.
Otherwise write two read-only DuckDB SELECTs (WITH allowed), each in its own ```sql``` block, orders first.

First SELECT, one row per order, with exactly these columns:
  order_id      the order / invoice id (one id = one order in the whole data)
  order_date    DATE the order was placed
  customer_id   customer code when the shop records one, NULL otherwise and for walk-ins
  customer_name customer name; NULL for walk-in customers (khách lẻ, kl, lẻ, vãng lai, blank), matched in any case/spacing
  total_due     what the customer must pay for the whole order, as the RULEBOOK defines the debt of an order. If the book only lists orders still owing after the purchase, this is the amount owing right after the purchase
  is_cancelled  BOOLEAN, the order is cancelled per the rulebook
  no_debt       BOOLEAN, the rulebook keeps this order out of debts whatever is unpaid (read the rulebook: e.g. a payment method or sales channel it excludes)
  is_credit     BOOLEAN, the order is recorded as sold on credit (the payment method the rulebook means by ghi nợ)

Second SELECT, one row per payment a customer made for one order, with exactly these columns:
  order_id      the order the payment is for, same id as the first SELECT
  pay_date      DATE of the payment; NULL only if the shop keeps a paid amount without any date
  amount        the money paid in that payment, positive

Rules: use only tables/columns listed in SCHEMA, always double-quoted. Keep cancelled and excluded orders: flag them, do not filter them out. Line tables must be reduced to one row per order. Do not take a stored "still owing" column as total_due when payments are recorded: debts at earlier days are computed from the payments.
Think briefly, then answer with the two ```sql``` blocks."""

CUSTOMER = "coalesce(customer_id, nullif(regexp_replace(replace(strip_accents(lower(customer_name)), 'đ', 'd'), '[^a-z0-9]', '', 'g'), ''))"


def parse(text):
    """(orders SELECT, payments SELECT), or None for a shop without debts."""
    text = text.split("</think>")[-1]
    blocks = [s.strip().rstrip(";") for s in re.findall(r"```(?:sql)?\s*(.*?)```", text, re.S)]
    if not blocks and re.fullmatch(r"\W*NONE\W*", text.strip()):
        return None
    if len(blocks) != 2:
        raise ValueError(f"need exactly two ```sql``` blocks, orders then payments; got {len(blocks)}")
    for s in blocks:
        if not re.match(r"(?is)\s*(with|select)\b", s) or re.search(r"(?i)\b(insert|update|delete|create|drop|alter|copy|attach|install|load|pragma)\b", s):
            raise ValueError("each block needs one read-only SELECT")
    return tuple(blocks)


def build(db, sqls, prefix="v_debt_"):
    """Materialize the tables and the debt_at macro from the two SELECTs; return a fingerprint, raise if unusable."""
    o, p = sqls
    db.execute(f"""CREATE OR REPLACE TABLE {prefix}orders_raw AS WITH src AS ({o}) SELECT CAST(order_id AS VARCHAR) AS order_id, CAST(order_date AS DATE) AS order_date,
        nullif(trim(CAST(customer_id AS VARCHAR)), '') AS customer_id, nullif(trim(CAST(customer_name AS VARCHAR)), '') AS customer_name, CAST(total_due AS DOUBLE) AS total_due,
        coalesce(CAST(is_cancelled AS BOOLEAN), false) AS is_cancelled, coalesce(CAST(no_debt AS BOOLEAN), false) AS no_debt, coalesce(CAST(is_credit AS BOOLEAN), false) AS is_credit FROM src""")
    db.execute(f"""CREATE OR REPLACE TABLE {prefix}pay_raw AS WITH src AS ({p}) SELECT CAST(order_id AS VARCHAR) AS order_id, CAST(pay_date AS DATE) AS pay_date, CAST(amount AS DOUBLE) AS amount
        FROM src WHERE amount IS NOT NULL AND amount <> 0""")
    n, ids, dated, bad_total = db.execute(f"SELECT count(*), count(DISTINCT order_id), avg(CAST(order_date IS NOT NULL AS INT)), count(*) FILTER (WHERE total_due IS NULL OR total_due < 0) FROM {prefix}orders_raw").fetchone()
    if not n or dated < 0.95:
        raise ValueError(f"unusable orders SELECT: rows={n}, share of rows with order_date={dated or 0:.2f}")
    if n - ids > 0.01 * n:
        raise ValueError(f"order_id is not unique: {n} rows for {ids} ids; the first SELECT needs one row per order")
    if bad_total > 0.01 * n:
        raise ValueError(f"total_due is NULL or negative on {bad_total} of {n} orders")
    np, orphan, neg = db.execute(f"""SELECT count(*), count(*) FILTER (WHERE order_id NOT IN (SELECT order_id FROM {prefix}orders_raw)), count(*) FILTER (WHERE amount < 0) FROM {prefix}pay_raw""").fetchone()
    if not np:
        raise ValueError("the payments SELECT returned no row: without payments nobody's debt can be computed")
    if orphan > 0.05 * np:
        raise ValueError(f"{orphan} of {np} payments have an order_id missing from the orders SELECT: use the same id in both")
    if neg > 0.05 * np:
        raise ValueError(f"{neg} of {np} payments are negative: amount is the money paid, positive")
    db.execute(f"""CREATE OR REPLACE TABLE {prefix}orders AS
        WITH paid AS (SELECT p.order_id, sum(amount) AS paid_total, coalesce(sum(amount) FILTER (WHERE p.pay_date IS NULL OR p.pay_date <= o.order_date), 0) AS paid_at_purchase
                      FROM {prefix}pay_raw p JOIN {prefix}orders_raw o USING (order_id) GROUP BY 1)
        SELECT o.order_id, o.order_date, {CUSTOMER} AS customer_key, o.customer_name, o.total_due AS order_total, o.is_cancelled, o.no_debt, o.is_credit,
               coalesce(paid.paid_at_purchase, 0) AS paid_at_purchase, o.total_due - coalesce(paid.paid_at_purchase, 0) AS new_debt
        FROM {prefix}orders_raw o LEFT JOIN paid USING (order_id)""")
    db.execute(f"""CREATE OR REPLACE TABLE {prefix}payments AS SELECT p.order_id, p.pay_date, p.amount, o.customer_key, o.customer_name, o.order_date,
        p.pay_date > o.order_date AS after_purchase, o.is_credit AND NOT o.is_cancelled AS is_credit
        FROM {prefix}pay_raw p JOIN {prefix}orders o USING (order_id)""")
    macro = "debt_at" if prefix == "v_debt_" else f"{prefix}at"
    db.execute(f"""CREATE OR REPLACE MACRO {macro}(x) AS TABLE
        WITH paid AS (SELECT order_id, sum(amount) AS paid FROM {prefix}pay_raw WHERE pay_date IS NULL OR pay_date <= x GROUP BY 1)
        SELECT o.order_id, o.order_date, o.customer_key, o.customer_name, o.order_total - coalesce(paid.paid, 0) AS debt, date_diff('day', o.order_date, CAST(x AS DATE)) AS age_days
        FROM {prefix}orders o LEFT JOIN paid USING (order_id)
        WHERE o.order_date <= x AND NOT o.is_cancelled AND NOT o.no_debt AND o.order_total - coalesce(paid.paid, 0) > 0""")
    if prefix == "v_debt_":  # what the prompt offers: macros only, a stored order total invites summing it instead of the debt
        db.execute(f"""CREATE OR REPLACE MACRO debt_new(a, b) AS TABLE SELECT order_id, order_date, customer_key, customer_name, new_debt FROM {prefix}orders
            WHERE is_credit AND NOT is_cancelled AND order_date >= a AND order_date < b""")
        db.execute(f"""CREATE OR REPLACE MACRO debt_repaid(a, b) AS TABLE SELECT order_id, pay_date, customer_key, customer_name, amount FROM {prefix}payments
            WHERE after_purchase AND is_credit AND pay_date >= a AND pay_date < b""")
    last = db.execute(f"SELECT max(order_date) FROM {prefix}orders").fetchone()[0]
    over = db.execute(f"""SELECT count(*) FROM (SELECT o.order_id FROM {prefix}orders o JOIN {prefix}pay_raw p USING (order_id) GROUP BY o.order_id, o.order_total HAVING sum(p.amount) > o.order_total + 1)""").fetchone()[0]
    if over > 0.05 * ids:
        raise ValueError(f"{over} of {ids} orders are paid more than total_due: total_due or the payment join is wrong")
    owed, owing = db.execute(f"SELECT sum(debt), count(*) FROM {macro}(DATE '{last}')").fetchone()
    if not owing:
        raise ValueError(f"no order owes anything at the last order date {last}: check total_due, no_debt and the payments")
    return n, round(db.execute(f"SELECT sum(order_total) FROM {prefix}orders").fetchone()[0]), np, round(db.execute(f"SELECT sum(amount) FROM {prefix}pay_raw").fetchone()[0]), owing, round(owed)


def check(db, texts):
    """Execute every draft into scratch tables -> [(fingerprint or 'none', sqls) or (None, error)]."""
    out = []
    for t in texts:
        timer = threading.Timer(60, db.interrupt)
        timer.start()
        try:
            sqls = parse(t)
            out.append(("none", None) if sqls is None else (build(db, sqls, "_dcand_"), sqls))
        except Exception as e:
            out.append((None, str(e)[:300]))
        finally:
            timer.cancel()
    for n in ("orders_raw", "pay_raw", "orders", "payments"):
        db.execute(f"DROP TABLE IF EXISTS _dcand_{n}")
    db.execute("DROP MACRO TABLE IF EXISTS _dcand_at")
    return out


def pick(checked):
    """(two SELECTs of the most common fingerprint, its count); (None, count) when most drafts say the shop has no debts."""
    fps = {}
    for fp, sqls in checked:
        if fp:
            fps.setdefault(fp, []).append(sqls)
    if not fps:
        return None, 0
    best = max(fps.values(), key=len)
    return best[0], len(best)


def accept(checked, n, k):
    """Use the tables when enough drafts agree, or when one set of numbers is all that passed the data tests (failed drafts are no vote against it)."""
    fps = {str(c) for c in checked if str(c) not in ("None", "none")}
    return n * 4 >= k or (n >= 1 and len(fps) == 1 and "none" not in {str(c) for c in checked})


def describe(n, k):
    return ("""DEBTS (%d/%d independent drafts agree; these table macros apply the RULEBOOK's debt rules once). Debt questions are answered ONLY through them, never from raw order or payment columns, because what is owed changes with every payment. X = the day asked, today when none is named; [A, B) = the period, B excluded.
debt_at(DATE 'X') = one row per order still owing at the end of day X: order_id, order_date, customer_key, customer_name, debt, age_days (days since the order date).
- total owed, TYPE number: SELECT ROUND(SUM(debt)) FROM debt_at(DATE 'X')
- customers owing, TYPE number: SELECT COUNT(DISTINCT customer_key) FROM debt_at(DATE 'X')
- top N customers and what each owes, TYPE table: SELECT customer_name, ROUND(SUM(debt)) AS so_tien_no FROM debt_at(DATE 'X') WHERE customer_key IS NOT NULL GROUP BY customer_key, customer_name ORDER BY 2 DESC, customer_key LIMIT N
- what one customer owes, TYPE number: SELECT ROUND(SUM(debt)) FROM debt_at(DATE 'X') WHERE customer_name = 'the name in SCHEMA spelling'
- debt older than D days, TYPE number: SELECT ROUND(SUM(debt)) FROM debt_at(DATE 'X') WHERE age_days > D
- debt split into age bands, TYPE list, one row in the order asked: SELECT ROUND(SUM(debt) FILTER (WHERE age_days BETWEEN 0 AND 7)), ROUND(SUM(debt) FILTER (WHERE age_days BETWEEN 8 AND 15)), ROUND(SUM(debt) FILTER (WHERE age_days > 15)) FROM debt_at(DATE 'X')
debt_new(DATE 'A', DATE 'B') = credit orders placed in the period with what stayed unpaid on the purchase day: order_id, order_date, customer_key, customer_name, new_debt. debt_repaid(DATE 'A', DATE 'B') = payments made in the period, after the purchase day, for credit orders: order_id, pay_date, customer_key, customer_name, amount.
- new debt, TYPE number: SELECT ROUND(SUM(new_debt)) FROM debt_new(DATE 'A', DATE 'B')
- repaid, TYPE number: SELECT ROUND(SUM(amount)) FROM debt_repaid(DATE 'A', DATE 'B')
- both for one customer, TYPE list in the order asked: SELECT (SELECT ROUND(SUM(new_debt)) FROM debt_new(DATE 'A', DATE 'B') WHERE customer_name = '...'), (SELECT ROUND(SUM(amount)) FROM debt_repaid(DATE 'A', DATE 'B') WHERE customer_name = '...')""" % (n, k))
