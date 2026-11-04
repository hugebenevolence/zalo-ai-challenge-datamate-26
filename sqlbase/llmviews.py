"""Per-shop semantic layer written by the LLM (no hand-written role/keyword rules).

The model reads one shop's README + SCHEMA + RULEBOOK and writes ONE SELECT that maps the raw tables to fixed
line-level columns. Generic SQL below casts them, lifts cancel/return flags to the whole order (rulebook: an order
with any cancelled/returned line is excluded as a whole) and derives v_sales / v_orders (valid only) plus *_all
variants. All samples are executed; the most common result fingerprint wins (execution-based self-consistency).
"""
import re
import threading
from collections import Counter

VIEW_SYSTEM = """You map ONE shop's raw tables (DuckDB) to a canonical sales table. Read README, SCHEMA and RULEBOOK carefully: they say which table holds the sales, which column is the order id, the date, the quantity, the line amount, the status.
Write ONE read-only DuckDB SELECT (WITH allowed) returning one row per sold line (one row per order or ticket if the shop records no lines), with exactly these columns:
  order_id      the order / invoice / ticket id (all lines of one order share it; one id = one order in the whole data: when the id is a running number that restarts per day/sheet/file, combine it with the date, e.g. strftime(date, '%Y%m%d') || '-' || id)
  order_date    DATE the order was placed (rulebook: the order date decides the period)
  order_ts      TIMESTAMP of the order, NULL if no time of day is recorded
  product_code  item code of the line, NULL if none
  product_group product group / category of the item (nhóm hàng, loại, danh mục) from the line or the catalog, NULL if none
  product_name  item / service name of the line (full name from a product catalog when the line only has a code or abbreviation), NULL if none
  qty           quantity sold on the line, as recorded
  unit          unit of qty, NULL if none
  amount        line revenue in VND after line-level discounts, before order-level vouchers, without VAT/shipping
  customer_id   customer code / member card id when the shop records one (mã khách, mã thẻ ...), NULL otherwise and for walk-ins
  customer_name customer name; NULL for walk-in customers (khách lẻ, kl, lẻ, vãng lai, blank), matched in any case/spacing: lower(trim(x)) IN ('khách lẻ', ...)
  staff, channel, payment   NULL when the shop does not record them
  is_cancelled  BOOLEAN, this line/order is cancelled per the rulebook (hủy/huỷ, không giao được ...)
  is_returned   BOOLEAN, this line/order is returned per the rulebook (trả hàng, đổi trả, hoàn, khách trả lại ..., also partial)
  line_kind     'item' (goods), 'service' (a fee line the rulebook counts in revenue, e.g. labour, service charge / phí phục vụ) or 'adjust' (voucher, points, discount line, shipping fee and other lines the rulebook keeps out of revenue); decide each fee by the rulebook, not by its name
Rules: use only tables/columns listed in SCHEMA, always double-quoted. Join header and line tables only on the order id. Join a catalog on every way a line can name the item, since one book often mixes them: its code or abbreviation, and its full name, compared as lower(trim(regexp_replace(x, '\\s+', ' ', 'g'))), never duplicating lines, and keep the line's own name/code when the catalog has no match: coalesce(catalog name, line name). A name written with a placeholder in the README ('Dùng <n> điểm') is a pattern: match it with LIKE ('Dùng % điểm'), never literally. Read status from the status column, or from a note column when there is none; check every listed status VALUE and classify each one. Keep every row: do NOT filter out cancelled, returned or adjust lines; flag them. If the amount is not stored, compute it (qty * unit price - line discount).
Think briefly, then answer with the SQL in one ```sql``` block."""

COLS = """CAST(order_id AS VARCHAR) AS order_id, CAST(order_date AS DATE) AS order_date, CAST(order_ts AS TIMESTAMP) AS order_ts,
  CAST(product_code AS VARCHAR) AS product_code, CAST(product_group AS VARCHAR) AS product_group, trim(regexp_replace(CAST(product_name AS VARCHAR), '\\s+', ' ', 'g')) AS product_name, CAST(qty AS DOUBLE) AS qty,
  CAST(unit AS VARCHAR) AS unit, CAST(amount AS DOUBLE) AS amount, nullif(trim(CAST(customer_id AS VARCHAR)), '') AS customer_id, nullif(trim(CAST(customer_name AS VARCHAR)), '') AS customer_name,
  CAST(staff AS VARCHAR) AS staff, CAST(channel AS VARCHAR) AS channel, CAST(payment AS VARCHAR) AS payment,
  coalesce(CAST(is_cancelled AS BOOLEAN), false) AS line_cancelled, coalesce(CAST(is_returned AS BOOLEAN), false) AS line_returned,
  lower(trim(CAST(line_kind AS VARCHAR))) AS line_kind"""
REV = "line_kind IN ('item', 'service')"


def parse_select(text):
    text = text.split("</think>")[-1]
    s = re.findall(r"```(?:sql)?\s*(.*?)```", text, re.S)
    sql = s[-1].strip().rstrip(";") if s else ""
    if not re.match(r"(?is)\s*(with|select)\b", sql) or re.search(r"(?i)\b(insert|update|delete|create|drop|alter|copy|attach|install|load|pragma)\b", sql):
        raise ValueError("need one read-only SELECT in ```sql```")
    return sql


def build(db, sql, prefix="v_"):
    """Materialize the four tables from the model's SELECT; return a fingerprint, raise if unusable."""
    db.execute(f"""CREATE OR REPLACE TABLE {prefix}sales_all AS
        WITH src AS ({sql}), t0 AS (SELECT {COLS} FROM src),
        -- rulebook: one customer = one customer code, else one name up to spelling (case, accents, spaces, punctuation)
        t1 AS (SELECT * REPLACE (coalesce(customer_id, nullif(regexp_replace(replace(strip_accents(lower(customer_name)), 'đ', 'd'), '[^a-z0-9]', '', 'g'), '')) AS customer_id) FROM t0),
        t AS (SELECT * EXCLUDE (customer_id, customer_name), customer_id AS customer_key,
              CASE WHEN customer_id IS NOT NULL THEN mode(customer_name) OVER (PARTITION BY customer_id) END AS customer_name FROM t1)
        SELECT *, bool_or(line_cancelled) OVER w AS is_cancelled, bool_or(line_returned) OVER w AS is_returned,
               NOT bool_or(line_cancelled OR line_returned) OVER w AS is_valid
        FROM t WINDOW w AS (PARTITION BY order_id)""")
    db.execute(f"CREATE OR REPLACE TABLE {prefix}sales AS SELECT * FROM {prefix}sales_all WHERE is_valid AND {REV}")
    db.execute(f"""CREATE OR REPLACE TABLE {prefix}orders_all AS SELECT order_id, min(order_date) AS order_date, min(order_ts) AS order_ts,
        bool_or(is_cancelled) AS is_cancelled, bool_or(is_returned) AS is_returned, bool_and(is_valid) AS is_valid,
        any_value(customer_key) AS customer_key, any_value(customer_name) AS customer_name, any_value(staff) AS staff, any_value(channel) AS channel, any_value(payment) AS payment,
        coalesce(sum(amount) FILTER (WHERE {REV}), 0) AS revenue, count(*) FILTER (WHERE {REV}) AS n_lines
        FROM {prefix}sales_all GROUP BY order_id""")
    db.execute(f"CREATE OR REPLACE TABLE {prefix}orders AS SELECT * FROM {prefix}orders_all WHERE is_valid")
    rows, orders, rev, ncan, nret, dated, bad = db.execute(f"""SELECT count(*), count(DISTINCT order_id),
        round(sum(amount) FILTER (WHERE is_valid AND {REV})), count(DISTINCT order_id) FILTER (WHERE is_cancelled),
        count(DISTINCT order_id) FILTER (WHERE is_returned), avg(CAST(order_date IS NOT NULL AS INT)),
        count(*) FILTER (WHERE line_kind IS NULL OR line_kind NOT IN ('item', 'service', 'adjust')) FROM {prefix}sales_all""").fetchone()
    if not rows or not rev or rev <= 0 or dated < 0.95 or bad:
        raise ValueError(f"unusable view: rows={rows} revenue={rev} share of rows with order_date={dated:.2f} rows with invalid line_kind={bad}")
    dup = rows - db.execute(f"SELECT count(*) FROM (SELECT DISTINCT * FROM {prefix}sales_all)").fetchone()[0]
    if dup > 0.01 * rows:  # identical lines were already merged at load (rulebook), so copies here come from a join fan-out
        raise ValueError(f"{dup} lines are exact copies: a join duplicates lines; join each lookup table on a unique key (one row per key)")
    nul = db.execute(f"SELECT avg(CAST(product_name IS NULL AS INT)) FROM {prefix}sales_all WHERE line_kind = 'item'").fetchone()[0] or 0
    if 0.02 < nul < 1:  # a catalog join that drops unmatched names loses their sales from every per-product question
        raise ValueError(f"product_name is NULL on {nul:.0%} of item lines: keep the line's own name when the catalog has no match")
    multi = db.execute(f"SELECT avg(CAST(n > 1 AS INT)) FROM (SELECT count(DISTINCT order_date) AS n FROM {prefix}sales_all GROUP BY order_id)").fetchone()[0]
    if multi > 0.02:  # one order has one date: else order_id merges different orders
        raise ValueError(f"order_id is not unique per order: {multi:.0%} of order_ids span several dates; build a unique id")
    adj = db.execute(f"""SELECT count(*) FROM {prefix}sales_all WHERE is_valid AND line_kind = 'adjust'
        AND product_name IN (SELECT product_name FROM {prefix}sales_all WHERE line_kind = 'item')""").fetchone()[0]
    if adj > 0.005 * rows:  # an 'adjust' line named like a sold product is that product's sale (e.g. one with a line discount)
        raise ValueError(f"{adj} lines of valid orders are 'adjust' but carry a product name that is sold as an item: "
                         "adjust is only for voucher / points / order-discount / non-revenue fee lines, a discounted product line is an item")
    top = db.execute(f"""SELECT any_value(customer_name), count(*) AS n FROM {prefix}orders WHERE customer_key IS NOT NULL
        GROUP BY customer_key ORDER BY n DESC LIMIT 2""").fetchall()
    nord = db.execute(f"SELECT count(*) FROM {prefix}orders").fetchone()[0]
    if top and top[0][1] > 0.05 * nord and top[0][1] > 10 * (top[1][1] if len(top) > 1 else 0):  # one 'customer' on a large share of orders is a placeholder
        raise ValueError(f"customer_name '{top[0][0]}' is on {top[0][1] / nord:.0%} of valid orders, over 10x any other customer: that is the walk-in label; "
                         "map every spelling of it to NULL, comparing lower(trim(name))")
    return rows, orders, rev, ncan, nret


_raw = {}


def raw_values(db):
    """Every text value of the shop's own tables, lower-cased (not the views or scratch tables): what a mapped column may hold."""
    cols = tuple(db.execute("SELECT table_name, column_name FROM information_schema.columns WHERE data_type = 'VARCHAR' AND table_name <> 'calendar' "
                            "AND NOT starts_with(table_name, 'v_') AND NOT starts_with(table_name, '_cand_') ORDER BY 1, 2").fetchall())
    key = (id(db), cols)  # the column list too: a closed connection's id can come back for another shop
    if key not in _raw:
        _raw[key] = {str(x[0]).strip().lower() for tab, col in cols
                     for x in db.execute(f'SELECT DISTINCT "{col}" FROM "{tab}" WHERE "{col}" IS NOT NULL LIMIT 50000').fetchall()}
    return _raw[key]


def catalog_test(db, prefix):
    """Drafting only (a reused view is not re-judged): a product_group constant no table holds ('khác', 'không rõ'), or a
    NULL group on part of the lines, stands for the lines the catalog join missed, so every per-group question loses them."""
    rows = db.execute(f"SELECT count(*) FROM {prefix}sales_all").fetchone()[0]
    made = [(g, n) for g, n in db.execute(f"SELECT product_group, count(*) FROM {prefix}sales_all WHERE line_kind = 'item' AND product_group IS NOT NULL "
                                          "GROUP BY 1 ORDER BY 2 DESC").fetchall() if g.strip().lower() not in raw_values(db)]
    if sum(n for _, n in made) > 0.02 * rows:
        raise ValueError(f"product_group '{made[0][0]}' is on {made[0][1]} lines but is no value of any table: those lines did not match the "
                         "catalog; match it on every way a line names the item (code or abbreviation, and full name)")
    nul = db.execute(f"SELECT avg(CAST(product_group IS NULL AS INT)) FROM {prefix}sales_all WHERE line_kind = 'item'").fetchone()[0] or 0
    if 0.02 < nul < 1:  # some lines have a group, others none: the same missed join, written as NULL
        raise ValueError(f"product_group is NULL on {nul:.0%} of item lines while the other lines have one: those lines did not match the "
                         "catalog; match it on every way a line names the item (code or abbreviation, and full name)")


def check(db, texts):
    """Execute every draft into scratch tables -> [(fingerprint, sql) or (None, error)]."""
    out = []
    for t in texts:
        timer = threading.Timer(60, db.interrupt)  # a draft that runs for a minute is a join blow-up
        timer.start()
        try:
            sql = parse_select(t)
            fp = build(db, sql, "_cand_")
            catalog_test(db, "_cand_")
            out.append((fp, sql))
        except Exception as e:
            out.append((None, str(e)[:300]))
        finally:
            timer.cancel()
    for n in ("sales_all", "sales", "orders_all", "orders"):
        db.execute(f"DROP TABLE IF EXISTS _cand_{n}")
    return out


def pick(db, checked):
    """Keep the SELECT of the most common fingerprint (ties: first) and build the views from it."""
    fps = {}
    for fp, sql in checked:
        if fp:
            fps.setdefault(fp, []).append(sql)
    if not fps:
        return None, 0
    sqls = max(fps.values(), key=len)
    fill = {}
    for s in dict.fromkeys(sqls):  # same numbers: prefer the draft whose catalog join fills the most item lines
        build(db, s)
        fill[s] = db.execute("SELECT count(product_group) + count(product_code) FROM v_sales_all WHERE line_kind = 'item'").fetchone()[0]
    best = max(fill, key=fill.get)  # ties: first
    build(db, best)
    return best, len(sqls)


def describe(db, sql, n, k):
    nv, no = db.execute("SELECT (SELECT count(*) FROM v_sales), (SELECT count(*) FROM v_orders)").fetchone()
    cols = ("order_ts", "product_code", "product_group", "product_name", "qty", "unit", "customer_key", "customer_name", "staff", "channel", "payment")
    filled = db.execute("SELECT " + ", ".join(f"count({c})" for c in cols) + " FROM v_sales_all").fetchone()
    empty = [c for c, x in zip(cols, filled) if not x]
    return (f'"v_sales" ({nv} rows) = item/service lines of VALID orders only (not cancelled, no returned line): revenue = SUM(amount), no status filter needed.\n'
            f'"v_orders" ({no} rows) = VALID orders only, one row per order: order_id, order_date, order_ts, customer_key, customer_name, staff, channel, payment, revenue, n_lines.\n'
            '"v_sales_all" / "v_orders_all" = every line / order with is_cancelled, is_returned, is_valid, line_kind (item|service|adjust).\n'
            "v_sales columns: order_id, order_date, order_ts, product_code, product_group, product_name, qty, unit, amount, customer_key, customer_name, staff, channel, payment, line_kind.\n"
            "Customers (rulebook): one customer = one customer_key (the customer code, else the name ignoring case/accents/spaces/punctuation); customer_name = its usual spelling; "
            "both NULL for walk-ins, who are no customer. Count customers with COUNT(DISTINCT customer_key), rank them GROUP BY customer_key and answer with customer_name."
            + (f" Always NULL here: {', '.join(empty)}." if empty else "")
            + f"\nBuilt from the raw tables by this mapping ({n}/{k} independent drafts agree):\n{sql}")
