# Adapted from data-agent-ver1 (MIT), salesview.py/roles.py
"""Rulebook-aware canonical sales views over a loaded business (public_test/rulebook.md).

add_views(db, meta) -> str creates two DuckDB tables next to the raw ones:
  v_sales   one row per sold line: order_id, order_date (DATE), order_ts (TIMESTAMP or NULL),
            status, is_cancelled, is_returned, is_valid, product_code, product_name, product_group,
            line_kind (item|service|adjust), unit, qty, factor, qty_base, base_unit, unit_price,
            amount, cost_unit, customer_key, customer_code, customer_name, customer_group, payment,
            channel, staff
  v_orders  one row per order: the order-level columns + revenue, n_lines
Semantics: an order with ANY returned line is a returned order and drops out as a whole; revenue =
sum of item/service line amounts (vouchers, points, shipping are adjust lines, not revenue); walk-in
customers have no customer_key; identical rows of one order count once.
Returns the text for the LLM prompt ("" when no sales table is found or the build fails).
"""
from __future__ import annotations

import re
import sys
import unicodedata

import numpy as np
import pandas as pd

STATUS_LEX = [  # (status, folded phrases), checked in this order
    ("tra_hang", ["tra hang", "doi tra", "hang tra ve", "tra lai", "khach tra", "khach mang tra", "return", "returned",
                  "refund", "refunded", "hoan tra", "hoan hang", "da hoan hang", "da tra hang", "tra ve", "doi hang"]),
    ("huy", ["huy", "cancel", "cancelled", "canceled", "void", "bo don", "khach bo", "khong giao duoc", "that bai",
             "failed"]),
    ("dang_giao", ["dang giao", "chua giao", "shipping", "delivering", "dang van chuyen", "cho giao", "in transit",
                   "pending", "dang ship", "dang gui", "cho lay hang", "dang giao dich", "dang xu ly", "processing"]),
    ("hoan_thanh", ["hoan thanh", "xong", "da lay", "completed", "complete", "done", "ok", "thanh cong", "da giao",
                    "success", "da thanh toan", "paid", "finished", "giao thanh cong", "hoan tat", "da ban",
                    "delivered", "closed", "received"]),
]
WALKIN = {"khach le", "kl", "le", "khach vang lai", "vang lai", "khach", "walk in", "walkin", "guest", "khach ghe",
          "khong ten", "khach qua duong", "na", "none"}
ADJUST_RX = re.compile(r"^(voucher|dung \d* ?diem|diem|phi ship|phi giao( hang)?|giam gia don|khuyen mai don|tru diem)\b")
SERVICE_RX = re.compile(r"^phi phuc vu")
FACTOR_RX = re.compile(r"^\s*\d+([.,]\d+)?(\s+[^\d=;,.]{1,15})?\s*$")   # "12", "15 vỉ"; not "1 thùng = 4 bịch"
CONV_RX = re.compile(r"1\s+([^\d=;,]+?)\s*=\s*([\d.,]+)\s+([^\d;,.]+)")
UNION_RX = re.compile(r"^union of \d+ tables:", re.I)
FILE_RX = r"([^\s,:#`]+)\.(?:xlsx|xls|csv|tsv|txt|db|sqlite3?)\b"
ORDER_LEVEL = {"order_date", "order_datetime", "order_time", "status", "note", "customer_name", "customer_code",
               "payment", "channel", "staff", "customer_group", "order_total", "order_discount", "paid", "debt",
               "shipping_fee", "order_goods_total"}

# (phrase in folded README description, role), most specific first; roles.py DESC_RULES
DESC_RULES = [
    ("so luong da doi ra don vi co ban", "qty_base"), ("so luong quy ve don vi co ban", "qty_base"),
    ("so luong ban lay tu lo", "lot_qty"), ("so luong khach tra lai", "returned_qty"),
    ("so luong nhap", "import_qty"), ("so luong giao", "delivery_qty"), ("so luong ton", "stock"),
    ("so thu tu dong", "line_no"), ("so thu tu don", "order_id"), ("so hoa don", "order_id"),
    ("ma don", "order_id"), ("ma hoa don", "order_id"), ("ma chung tu", "order_id"),
    ("ma phieu nhap", "receipt_id"), ("ma phieu thu", "payment_id"),
    ("ma nha cung cap", "supplier"), ("nha cung cap", "supplier"),
    ("ma khach", "customer_code"), ("ten khach", "customer_name"), ("nhom khach", "customer_group"),
    ("ma nhan vien", "staff"), ("nhan vien", "staff"), ("tho lam", "staff"),
    ("ma hang", "product_code"), ("ma san pham", "product_code"), ("ma thuoc", "product_code"),
    ("ten hang", "product_name"), ("hang ban", "product_name"), ("ten thuoc", "product_name"),
    ("ten san pham", "product_name"), ("ten mon", "product_name"),
    ("nhom hang", "product_group"), ("loai hang", "product_group"),
    ("don vi co ban", "base_unit"), ("don vi lon", "big_unit"), ("don vi nhap", "import_unit"),
    ("don vi tinh", "unit"), ("he so", "factor"),
    ("thoi diem dat don", "order_datetime"), ("gio dat don", "order_time"),
    ("ngay ban", "order_date"), ("ngay dat", "order_date"), ("ngay mua", "order_date"),
    ("ngay nhap", "import_date"), ("ngay giao", "delivery_date"), ("ngay sua", "order_date"),
    ("han su dung", "expiry"), ("so lo", "lot"),
    ("trang thai phieu nhap", "import_status"), ("trang thai don", "status"), ("trang thai", "status"),
    ("thanh tien sau giam", "line_amount"), ("thanh tien cua dong", "line_amount"),
    ("tien nhap cua ca dong", "import_amount"), ("tong tien phieu nhap", "import_total"),
    ("tien giam cua ca dong", "line_discount"),
    ("giam gia cua ca don", "order_discount"), ("tien bot cho ca don", "order_discount"),
    ("tong tien hang cua ca don", "order_goods_total"), ("tong tien hang cua don", "order_goods_total"),
    ("tien hang cua don", "order_goods_total"),
    ("so khach phai tra", "order_total"), ("khach can tra", "order_total"), ("tong tien don", "order_total"),
    ("tong tien cua ca phieu", "order_total"), ("tong tien ca phieu", "order_total"),
    ("so tien khach tra cho ca don", "order_total"),
    ("phi giao", "shipping_fee"),
    ("khach da tra", "paid"), ("da thanh toan", "paid"),
    ("khach con no", "debt"), ("con no", "debt"),
    ("hinh thuc thanh toan", "payment"), ("kenh ban", "channel"),
    ("gia nhap", "import_price"), ("gia von", "cost"), ("don gia", "unit_price"),
    ("gia ban hien tai", "list_price"), ("ton kho", "stock"),
    ("so dien thoai", "phone"), ("dia chi", "address"), ("ghi chu", "note"),
    ("giam gia", "discount"), ("thanh tien", "line_amount"), ("so luong", "qty"),
]
# (regex on folded header, role), evaluated in order; roles.py NAME_RULES
NAME_RULES = [
    (r"^(stt|so tt|tt dong|so thu tu|thu tu|line|no|dong|dong so)$", "line_no"),
    (r"(sl|so luong|slg)\s*((dv|don vi)\s*)?(quy doi|quy ra|co ban|goc|le|quy ra dv le|quy ra le)|quy ra (dv )?le|sl co ban|slquy doi|so luong le", "qty_base"),
    (r"sl tra|so luong tra|tra lai", "returned_qty"),
    (r"(sl|so luong) nhap", "import_qty"),
    (r"(sl|so luong) giao|thuc giao", "delivery_qty"),
    (r"^(ma (hoa don|hd|don|don hang|chung tu|giao dich|bill)|so (hd|hoa don|bill|don|chung tu)|(don|hd|hoa don|bill) so|order id|order no|ma bill|hd|bill|invoice|ma hoadon|ma don ban)$", "order_id"),
    (r"^(ma (phieu nhap|nhap hang|pn)|so phieu|ma phieu)$", "receipt_id"),
    (r"^(ma (kh|khach|khach hang)|so kh|customer id|ma kh mua|id khach)$", "customer_code"),
    (r"^(ten (kh|khach|khach hang)|(khach hang|khach|kh) ten|khach hang|khach|ben mua|doi tac|customer|nguoi mua|ten khach mua|customer name)$", "customer_name"),
    (r"(nhom|loai) khach|hang khach|khach le si|loai khach hang", "customer_group"),
    (r"^(ma (nv|nhan vien)|nhan vien|nv ban|nguoi ban|thu ngan|ma nv ban|tho)$", "staff"),
    (r"^(ma (hang|sp|san pham|hh|hang hoa|mon|thuoc|vach)|id sp|sku|ma hanghoa|ma|(item|product|sku) (code|id)|product id|item no)$", "product_code"),
    (r"^(ten (hang|sp|san pham|hang hoa|mon|thuoc|mat hang)|hang|hang hoa|mat hang|san pham|mon|noi dung|chi tiet hang|item|thuoc|ten|ten goi|ten day du|(item|product) name|product|description)$", "product_name"),
    (r"nhom (hang|hh|hang hoa|sp)|danh muc|phan loai|nhom|loai$|category", "product_group"),
    (r"(don vi|dv|dvt)\s*(goc|co ban|co so)|don vi co so", "base_unit"),
    (r"^(dv|don vi) (nhap|lon|quy doi|doi|quy cach)$|^dvqd$|^don vi quy doi( lon)?$|^quy cach$", "big_unit"),
    (r"^(dvt|don vi( tinh)?|dv|d vi|unit|dv tinh)$", "unit"),
    (r"^(?!gia\b)(?!.*\bgia (ban|nhap)\b).*(he so|quy doi|\bbang\b)", "factor"),
    (r"^(gio|gio tao don|gio dat|time)$", "order_time"),
    (r"thoi (gian|diem) (lap|tao|dat)|gio tao don|created", "order_datetime"),
    (r"^(ngay( ban| dat| giao dich| lap| mua| thang| hd| tao)?|ngay ban hang|ngaylap|ngay ban le|ngay chung tu|thoi gian)$", "order_date"),
    (r"ngay nhap", "import_date"), (r"ngay giao", "delivery_date"),
    (r"han su dung|hsd|han dung|expiry", "expiry"), (r"^(lo|so lo|ma lo)$", "lot"),
    (r"trang thai|status|ket qua|xu ly|tinh trang", "status"),
    (r"^(thanh tien|t tien|tri gia|tt|tt dong|thanh tien dong|tien|thanh tien sau giam|thanhtien|amount|line (total|amount)|subtotal)$", "line_amount"),
    (r"^(giam|giam dong|ck dong|chiet khau dong)$", "line_discount"),
    (r"tien hang|tong tien hang", "order_goods_total"),
    (r"^(tong tien|tong bill|tong thanh toan|tong don|phai thu|tong sau giam gia|khach phai tra|can tra)$", "order_total"),
    (r"chiet khau|giam gia|khuyen mai|^ck$|^bot$", "discount"),
    (r"phi (giao|ship|van chuyen)", "shipping_fee"),
    (r"^(da tra|khach da tra|da thu|khach tra)$", "paid"),
    (r"con no|chua tra|cong no|con lai|con phai thu", "debt"),
    (r"phuong thuc|hinh thuc (tt|thanh toan)|thanh toan|payment", "payment"),
    (r"^(chi nhanh|ma chi nhanh|cua hang|co so|branch|store|diem ban)$", "branch"),
    (r"kenh|ban qua|nguon don|^nguon$|channel", "channel"),
    (r"gia von", "cost"), (r"gia nhap", "import_price"),
    (r"^(don gia|gia|gia ban|gia le|gia niem yet|gia don vi|don gia ban|dg|unit price|price|gia san pham|gia (1|mot) (dv|don vi))$", "unit_price"),
    (r"^(sl|s l|so luong|slg|qty|so luong ban|sl ban|quantity|qty sold|so luong san pham)$", "qty"),
    (r"ton (kho|cuoi|thuc)|^ton$", "stock"),
    (r"nha cung cap|ncc", "supplier"),
    (r"dien thoai|sdt|phone", "phone"), (r"dia chi|addr", "address"),
    (r"ghi chu|note|dien giai", "note"),
]
NUMERIC_ROLES = {"qty", "qty_base", "returned_qty", "import_qty", "delivery_qty", "lot_qty", "factor", "unit_price",
                 "line_amount", "line_discount", "discount", "order_discount", "order_goods_total", "order_total",
                 "shipping_fee", "paid", "debt", "cost", "import_price", "import_amount", "import_total", "stock",
                 "list_price"}
DATE_ROLES = {"order_date", "order_datetime", "import_date", "delivery_date", "expiry"}
TEXT_ROLES = {"order_id", "receipt_id", "payment_id", "customer_code", "customer_name", "product_code",
              "product_name", "product_group", "unit", "base_unit", "big_unit", "import_unit", "status", "note",
              "payment", "channel", "staff", "supplier", "customer_group", "lot", "phone", "address", "branch"}
# (kind, regex on folded table name / file names), first match wins; roles.py KIND_RULES with optional spaces
KIND_RULES = [
    ("purchase", r"nhap ?hang|phieu nhap|chi tiet nhap|nhap kho|so nhap|don nhap|^nhap|\bnhap\b"),
    ("stock", r"ton kho|bao cao ton|^ton\b|\bton\b|the kho"),
    ("price_history", r"bang gia|lich su gia"),
    ("lot", r"lo hang|ban theo lo|chi tiet lo|\blo\b"),
    ("payment", r"thanh toan|phieu thu|so quy|thu tien"),
    ("delivery", r"giao hang|phieu giao|so giao"),
    ("debt", r"cong no|han no|so no\b|tra gop|so gop"),
    ("expense", r"chi phi"),
    ("glossary", r"ghi tat|viet tat|tu dien|chu thich|giai thich"),
    ("warranty", r"bao hanh"),
    ("return", r"tra hang|hoan tra|doi tra|hang tra"),
    ("customer", r"khach ?hang|^khach|thanh vien"),
    ("supplier", r"nha cung cap|\bncc\b"),
    ("branch", r"chi nhanh|\bco so\b"),
    ("conversion", r"quy doi|don vi|\bdvt\b"),
    ("sales", r"hoa ?don|don ?hang|chi tiet don|chi tiet ban|bill"),
    ("product", r"san pham|hang ?hoa|danh ?muc|\bthuoc\b|\bmon\b|mat hang"),
    ("staff", r"nhan vien|\btho\b"),
    ("sales", r"ban ?hang|ban le|so ban|\bpos\b|du lieu ban|sua xe|giao dich|chi tiet"),
]


# ------------------------------------------------------------ helpers ---
def _na(v) -> bool:
    return v is None or v is pd.NA or v is pd.NaT or (isinstance(v, (float, np.floating)) and np.isnan(v))


def fold(s) -> str:
    """Accent/case/punctuation-insensitive key, words separated by single spaces (matching only)."""
    if _na(s):
        return ""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(s)))   # camelCase
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "d").lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def _mnorm(s) -> str:
    """Accent-preserving key: different letters or digits are different customers (rulebook section 3)."""
    s = unicodedata.normalize("NFC", str(s)).lower()
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in s).split())


def _sid(v):
    """Order / product code as a clean string (1.0 -> '1')."""
    if _na(v):
        return None
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    s = str(v).strip()
    return s or None


def _code(v):
    s = _sid(v)
    return None if s is None or not fold(s) or fold(s) in WALKIN else s


def norm_status(v):
    if _na(v):
        return None
    f = f" {fold(v)} "
    for st, phrases in STATUS_LEX:
        if any(f" {p} " in f for p in phrases):
            return st
    return None


def ckey(name):
    f = fold(name)
    if not f or f in WALKIN or f.startswith("khach le"):
        return None
    return re.sub(r"(?<=\d) (?=\d)", "", _mnorm(name))


def _tod(s: pd.Series) -> pd.Series:
    """Time of day as Timedelta from 'HH:MM[:SS]' text or datetime values."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return (s - s.dt.normalize()).fillna(pd.Timedelta(0))
    m = s.astype(str).str.extract(r"(\d{1,2}):(\d{2})(?::(\d{2}))?").astype(float).fillna(0)
    return pd.to_timedelta(m[0] * 3600 + m[1] * 60 + m[2], unit="s")


def _readme_sections(text):
    """[(folded title, {folded column: description})] for '## Cột trong `file`' sections (readme.py)."""
    marks = list(re.finditer(r"^##\s+(.*)$", text or "", re.M))
    out = []
    for i, m in enumerate(marks):
        t = fold(m.group(1))
        if not (t.startswith("cot trong") or t.startswith("cot cua") or "columns" in t):
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        cols = {}
        for c in re.finditer(r"^\s*[-*]\s+\*\*(.+?)\*\*\s*[:：]\s*(.+?)\s*$", text[m.end():end], re.M):
            cols.setdefault(fold(c.group(1)), c.group(2))
        out.append((t, cols))
    return out


def _desc(sections, source, col):
    """Description of `col` in the README section of the table's file (and sheet when named).
    Falls back to the one description the column has in any section (README file names may be patterns)."""
    sheets = [fold(s) for s in re.findall(r"#([^\s,]+)", source or "")]
    k = fold(col)
    for f in dict.fromkeys(fold(x) for x in re.findall(FILE_RX, source or "", re.I)):
        hits = [(any(f" {s} " in f" {t} " for s in sheets), cols) for t, cols in sections if f" {f} " in f" {t} "]
        for _, cols in sorted(hits, key=lambda h: -h[0]):
            if k in cols:
                return cols[k]
    anywhere = {cols[k] for _, cols in sections if k in cols}
    return anywhere.pop() if len(anywhere) == 1 else None


def _dtype_ok(role, s: pd.Series) -> bool:
    nn = s.dropna()
    if nn.empty:
        return False
    if role == "factor" and not pd.api.types.is_numeric_dtype(s):
        return nn.astype(str).str.match(FACTOR_RX).mean() > 0.9
    if role in NUMERIC_ROLES:
        return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if role in DATE_ROLES:
        return pd.api.types.is_datetime64_any_dtype(s)
    if role == "order_time":
        return pd.api.types.is_datetime64_any_dtype(s) or nn.astype(str).str.match(r"^\d{1,2}:\d{2}").mean() > 0.9
    if role in ("order_id", "line_no", "product_code", "customer_code"):
        return True   # codes may be numeric
    if role in TEXT_ROLES:
        return not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s)
    return True


def _role_of(s, col, desc):
    """Role of one column: README description first, then header name, dtype always checked."""
    if desc:
        d = f" {fold(desc)}"
        best = None   # the phrase the description starts with wins
        for phrase, role in DESC_RULES:
            i = d.find(f" {phrase}")
            if 0 <= i <= 12 and (best is None or (i, -len(phrase)) < (best[0], -best[2])):
                best = (i, role, len(phrase))
        if best and _dtype_ok(best[1], s):
            return best[1], "readme"
    k = fold(col)
    bare = re.sub(r"\s(d|vnd|dong|k)$", "", k)   # "Thành tiền (đ)"
    for kk in dict.fromkeys((k, bare)):
        for rx, role in NAME_RULES:
            if re.search(rx, kk) and _dtype_ok(role, s):
                return role, "header"
    return None, ""


def _kind(name, source, roles):
    r = set(roles.values())
    has_prod = bool(r & {"product_name", "product_code"})
    has_qty = bool(r & {"qty", "qty_base", "line_amount"})
    for text in (fold(name), fold(UNION_RX.sub("", source or ""))):
        for kind, rx in KIND_RULES:
            if re.search(rx, text):
                if kind == "product" and "order_id" in r and has_qty:
                    return "sales"
                if kind == "customer" and has_prod and has_qty:
                    return "sales"
                return kind
    if has_prod and has_qty and r & {"order_id", "order_date", "order_datetime"}:
        return "sales"
    if "order_id" in r and r & {"order_date", "order_datetime", "status"}:
        return "sales"
    if has_prod and not r & {"order_date", "order_datetime"}:
        return "product"
    return "other"


def _inv(t):
    """role -> first column with that role"""
    out = {}
    for c, r in t["roles"].items():
        out.setdefault(r, c)
    return out


def _load(db, meta):
    """Read every visible table, tag its column roles and kind (roles.tag_business)."""
    sections = _readme_sections(meta.get("readme", ""))
    T = {}
    for info in meta["tables"]:
        df = db.execute(f'SELECT * FROM "{info["name"]}"').df()
        roles, ev = {}, {}
        for c in df.columns:
            if str(c).startswith("_"):   # _sheet, _source, _huy ... are bookkeeping
                continue
            r, e = _role_of(df[c], str(c), _desc(sections, info["source"], str(c)))
            if r:
                roles[c], ev[c] = r, e
        if "order_id" not in roles.values():  # a repeated line number groups the lines of one order ("các dòng cùng STT là một đơn")
            for c, r in list(roles.items()):
                if r == "line_no" and df[c].notna().mean() > 0.95 and df[c].nunique() < 0.9 * df[c].notna().sum():
                    roles[c], ev[c] = "order_id", "repeated line number"
                    break
        T[info["name"]] = {"name": info["name"], "df": df, "src": info["source"], "roles": roles, "ev": ev}
    for t in T.values():
        t["kind"] = _kind(t["name"], t["src"], t["roles"])
    for t in T.values():     # "name" means different things in different lists
        for c, r in list(t["roles"].items()):
            if t["kind"] == "customer" and r in ("product_name", "product_code"):
                t["roles"][c] = "customer_name" if r == "product_name" else "customer_code"
            elif t["kind"] in ("staff", "supplier", "branch") and r in ("product_name", "product_code", "customer_name"):
                del t["roles"][c]
    return T


def _catalog(T):
    """Product catalog table (item master) -> (DataFrame[product_code, product_name, product_group, base_unit, cost], name)."""
    best = None
    for name, t in T.items():
        i = _inv(t)
        if t["kind"] not in ("product", "other", "stock", "conversion") or "product_name" not in i:
            continue
        if t["kind"] == "conversion" and "product_code" not in i:
            continue
        score = t["df"][i["product_name"]].nunique() + (1000 if "product_code" in i else 0) + \
            (500 if t["kind"] == "product" else 0)
        if best is None or score > best[0]:
            best = (score, name)
    if best is None:
        return None, None
    t = T[best[1]]
    i, df = _inv(t), t["df"]
    P = pd.DataFrame(index=df.index)
    P["product_code"] = df[i["product_code"]].map(_sid) if "product_code" in i else None
    P["product_name"] = df[i["product_name"]]
    P["product_group"] = df[i["product_group"]] if "product_group" in i else None
    P["base_unit"] = df[i["base_unit"]] if "base_unit" in i else (df[i["unit"]] if "unit" in i else None)
    P["cost"] = df[i["cost"]] if "cost" in i else None
    P = P.dropna(subset=["product_name"])
    P = P.drop_duplicates(subset=["product_code"] if P["product_code"].notna().any() else ["product_name"])
    return P, best[1]


def _cat(P, field, codes, names):
    """Catalog value per line: by product code when the line has one, else by folded product name."""
    cm = dict(zip(P["product_code"], P[field])) if P["product_code"].notna().any() else {}
    nm = {fold(n): v for n, v in zip(P["product_name"], P[field])}
    return [cm[c] if c in cm else nm.get(fold(n)) for c, n in zip(codes, names)]


def _conversions(T):
    """{(folded product, folded unit): factor to the base unit}, from factor columns and 'Quy đổi' text
    ('1 thùng = 4 bịch'); chained through intermediate units ('1 hộp = 15 vỉ', '1 vỉ = 12 viên')."""
    raw = {}
    for t in T.values():
        if t["kind"] not in ("conversion", "product", "other"):
            continue
        i, df = _inv(t), t["df"]
        prods = [i[r] for r in ("product_code", "product_name") if r in i]
        unit = next((i[r] for r in ("big_unit", "unit", "import_unit") if r in i), None)
        for p_col in prods:
            if "factor" in i and unit:
                for p, u, f in zip(df[p_col], df[unit], df[i["factor"]]):
                    fc = _factor_cell(f)
                    if not _na(p) and not _na(u) and fc:
                        raw[(fold(p), fold(u))] = fc
            for c in df.columns:
                if "quy doi" in fold(c) and not pd.api.types.is_numeric_dtype(df[c]):
                    for p, txt in zip(df[p_col], df[c]):
                        for m in CONV_RX.finditer(str(txt) if not _na(txt) else ""):
                            try:
                                raw[(fold(p), fold(m.group(1)))] = (float(m.group(2).replace(",", ".")), None)
                            except ValueError:
                                pass
    conv = {}
    for (p, u), (f, via) in raw.items():
        seen, total = {u}, f
        while via and via not in seen and (p, via) in raw:
            seen.add(via)
            f2, via = raw[(p, via)]
            total *= f2
        conv[(p, u)] = total
    return conv


def _factor_cell(f):
    """12 -> (12.0, None); '15 vỉ' -> (15.0, 'vi'); junk -> None."""
    if _na(f):
        return None
    if isinstance(f, (int, float, np.integer, np.floating)):
        return (float(f), None) if f > 0 else None
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*(\D.{0,15})?$", str(f))
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    via = fold(m.group(2)) if m.group(2) else ""
    return (v, via or None) if v > 0 else None


def _pick_header(T, lt, oids):
    """Order-header table (one row per order) that covers the line order ids (>50%)."""
    best = None
    for t in T.values():
        if t is lt or t["kind"] not in ("sales", "other", "debt", "payment"):
            continue
        i = _inv(t)
        if "order_id" not in i or not any(r in i for r in ORDER_LEVEL):
            continue
        right = t["df"][i["order_id"]].map(_sid).dropna()
        if not right.is_unique and t["kind"] != "sales":
            continue
        cov = pd.Series(list(oids)).isin(set(right)).mean() if len(oids) else 0.0
        score = cov * 10 + sum(r in i for r in ORDER_LEVEL) + (2 if t["kind"] == "sales" else 0)
        if cov > 0.5 and (best is None or score > best[0]):
            best = (score, t, i["order_id"])
    return None if best is None else (best[1], best[2])


def _orders_only(T):
    """Shops that keep one row per order and no item lines: v_orders only (revenue = order amount)."""
    money_roles = {"order_goods_total", "line_amount", "order_total"}
    cands = [t for t in T.values() if t["kind"] == "sales" and {"order_date", "order_datetime"} & set(t["roles"].values())
             and money_roles & set(t["roles"].values())]
    if not cands:
        return None
    t = max(cands, key=lambda t: ("order_id" in _inv(t), len(t["df"])))
    i, df = _inv(t), t["df"]
    n, warns = len(df), ["no item lines: v_orders only, no v_sales"]

    def S(r):
        return df[i[r]] if r in i else pd.Series([None] * n, index=df.index, dtype=object)
    if "order_id" in i:
        oid = S("order_id").map(_sid)
    else:
        oid = pd.Series([f"row{k}" for k in range(n)], index=df.index)
        warns.append("no order id: each row is one order")
    ts = pd.to_datetime(S("order_datetime") if "order_datetime" in i else S("order_date"), errors="coerce")
    tt = ts.dropna()
    has_time = len(tt) > 0 and (tt != tt.dt.normalize()).mean() > 0.3
    st = S("status").map(norm_status)
    st = st.where(st.notna(), S("note").map(norm_status)).fillna("hoan_thanh")
    money = next(r for r in ("order_goods_total", "line_amount", "order_total") if r in i)
    if money == "order_total":
        warns.append("revenue taken from the order total column (may include order discounts and fees)")
    ccode = S("customer_code").map(_code)
    O = pd.DataFrame({
        "order_id": oid, "order_date": ts.dt.normalize(),
        "order_ts": ts if has_time else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]"),
        "status": st, "is_cancelled": st.eq("huy"), "is_returned": st.eq("tra_hang"),
        "is_valid": ~st.isin(["huy", "tra_hang"]),
        "customer_key": ccode if "customer_code" in i else S("customer_name").map(ckey),
        "customer_code": ccode, "customer_name": S("customer_name"), "customer_group": S("customer_group"),
        "payment": S("payment"), "channel": S("channel"), "staff": S("staff"),
        "revenue": pd.to_numeric(S(money), errors="coerce"), "n_lines": pd.Series(np.nan, index=df.index),
    }).drop_duplicates("order_id").reset_index(drop=True)
    info = {"notes": [f'orders="{t["name"]}"', "order-level only"], "warnings": warns, "roles": i, "ev": t["ev"]}
    return None, O, info


# -------------------------------------------------------------- builder ---
def _build(T):
    """-> (v_sales, v_orders, info); v_sales is None for order-only shops; None when no sales table is found."""
    warns, notes = [], []
    lines = [t for t in T.values() if t["kind"] == "sales"
             and set(t["roles"].values()) & {"product_name", "product_code"}
             and set(t["roles"].values()) & {"qty", "qty_base", "line_amount"}]
    if not lines:
        return _orders_only(T)
    lt = max(lines, key=lambda t: len(t["df"]))
    I = _inv(lt)
    L = lt["df"].reset_index(drop=True)
    n = len(L)
    notes.append(f'lines="{lt["name"]}"')

    # order id: blank cells on follow-up rows inherit the order; orphan rows are dropped
    if "order_id" in I:
        ids = L[I["order_id"]].map(_sid)
        if ids.isna().any():
            prod = I.get("product_name") or I.get("product_code")
            if prod and L.loc[ids.isna(), prod].notna().mean() > 0.8:
                ids = ids.ffill()
                notes.append(f"{I['order_id']} forward-filled")
            keep = ids.notna()
            if (~keep).any():
                warns.append(f"{int((~keep).sum())} row(s) without order id dropped")
                L, ids = L[keep].reset_index(drop=True), ids[keep].reset_index(drop=True)
    else:
        ids = pd.Series([f"row{i}" for i in range(n)])
        warns.append("no order id column: each line treated as one order")
    L["__oid"] = ids.values
    data_cols = [c for c in L.columns if c == "__oid" or not str(c).startswith("_")]
    dup = L.duplicated(subset=data_cols, keep="first")   # same order, identical lines: export duplicate
    if dup.any():
        warns.append(f"{int(dup.sum())} duplicate line(s) of the same order removed")
        L = L[~dup].reset_index(drop=True)
    n = len(L)
    for c, r in lt["roles"].items():
        if r in ORDER_LEVEL and c in L.columns and L[c].isna().any():
            L[c] = L[c].fillna(L.groupby("__oid", sort=False)[c].transform("first"))

    # order-level facts missing from the line table come from the order-header table
    need = {"order_date", "order_datetime", "status", "note", "customer_code", "customer_name", "payment", "channel",
            "staff"} - set(I)
    hdr = _pick_header(T, lt, set(L["__oid"].dropna())) if need else None
    if hdr:
        ht, hoid = hdr
        hi = _inv(ht)
        take = {r: hi[r] for r in need if r in hi}
        if take:
            H = pd.DataFrame({"__oid": ht["df"][hoid].map(_sid).values})
            for r, c in take.items():
                H[f"__h_{r}"] = ht["df"][c].values
            L = L.merge(H.drop_duplicates("__oid"), on="__oid", how="left")
            for r in take:
                I[r] = f"__h_{r}"
            notes.append(f'header="{ht["name"]}" ({", ".join(take)})')

    def S(role):
        return L[I[role]] if role in I else pd.Series([None] * n, index=L.index, dtype=object)

    def num(role):
        return pd.to_numeric(S(role), errors="coerce")

    # dates
    ts = None
    if "order_datetime" in I:
        ts = pd.to_datetime(S("order_datetime"), errors="coerce")
    if "order_date" in I:
        dd = pd.to_datetime(S("order_date"), errors="coerce").dt.normalize()
        if ts is None:
            ts = dd + _tod(S("order_time")) if "order_time" in I else dd
        else:
            ts = ts.fillna(dd)
    if ts is None:
        raise ValueError("no order date column")
    oid = L["__oid"]

    # products
    P, cat_name = _catalog(T)
    codes = [_sid(v) for v in S("product_code")]
    names = [None if _na(v) else str(v).strip() for v in S("product_name")]
    group = list(S("product_group"))
    base = [None] * n
    if P is not None:
        names = [c if c is not None else x for c, x in zip(_cat(P, "product_name", codes, names), names)]
        codes = [c if c is not None else x for c, x in zip(codes, _cat(P, "product_code", codes, names))]
        cg, cb = _cat(P, "product_group", codes, names), _cat(P, "base_unit", codes, names)
        group = [g if not _na(g) else x for g, x in zip(group, cg)]
        base = cb
        notes.append(f'catalog="{cat_name}"')
    codes = [c if c is not None else x for c, x in zip(codes, names)]
    kind = ["adjust" if ADJUST_RX.match(fold(x)) else "service" if SERVICE_RX.match(fold(x)) else "item"
            for x in names]
    kind = pd.Series(kind, index=L.index)

    # quantities: qty_base from the file, else qty x factor from the conversion rules (unknown -> 1)
    q = num("qty") if "qty" in I else (num("qty_base") if "qty_base" in I else pd.Series(1.0, index=L.index))
    unit = S("unit")
    if "qty_base" in I:
        qbase = num("qty_base")
        factor = (qbase / q).where(q != 0)
    else:
        conv = _conversions(T)
        fac = []
        for c, x, u, b in zip(codes, names, unit, base):
            uf = fold(u)
            if not uf or uf == fold(b):
                fac.append(1.0)
            else:
                fac.append(conv.get((fold(c), uf)) or conv.get((fold(x), uf)) or 1.0)
        factor = pd.Series(fac, index=L.index, dtype=float)
        qbase = q * factor

    # money: line amount after line discount; qty x price - discount when the file has no amount
    price = num("unit_price")
    disc = num("line_discount")
    if "line_discount" not in I:
        disc = num("discount")
    if "line_amount" in I:
        amount = num("line_amount")
    elif "unit_price" in I:
        amount = q * price - disc.fillna(0)
        notes.append("amount = qty x unit_price - discount")
    else:
        amount = pd.Series(np.nan, index=L.index)
        warns.append("no line amount or price: revenue unavailable")

    # customers: a code column decides identity (walk-in codes are not customers); else the name
    cn = S("customer_name")
    if "customer_code" in I:
        ccode = S("customer_code").map(_code)
        ckeys = ccode
        ct = next((t for t in T.values() if t["kind"] == "customer" and {"customer_code", "customer_name"} <= set(t["roles"].values())), None)
        if ct is not None and cn.isna().all():
            ci = _inv(ct)
            m = dict(zip(ct["df"][ci["customer_code"]].map(_sid), ct["df"][ci["customer_name"]]))
            cn = ccode.map(m)
    else:
        ccode = pd.Series([None] * n, index=L.index, dtype=object)
        ckeys = cn.map(ckey)

    # status: STATUS_LEX on the status column, else on the note column
    st_a = S("status").map(norm_status)
    st_b = S("note").map(norm_status)
    st = st_a.where(st_a.notna(), st_b)
    if "status" not in I and "note" not in I:
        warns.append("no status column or note: every order treated as completed")
    elif "status" not in I:
        warns.append("no status column: status read from the note column")
    if "status" in I:
        unk = S("status").notna() & st_a.isna()
        if unk.any():
            warns.append(f"{unk.mean():.1%} lines with unrecognised status treated as completed: "
                         f"{sorted(set(map(str, S('status')[unk].unique())))[:5]}")
    ret_q = num("returned_qty").fillna(0) > 0 if "returned_qty" in I else pd.Series(False, index=L.index)
    neg = ((q < 0) | (amount < 0)).fillna(False) & kind.eq("item")
    ret = st.eq("tra_hang") | neg | ret_q
    # the loader's status flags (ingest STATUS_COL/CANCELLED/RETURNED) also mark cancelled / returned lines
    if "_huy" in L.columns:
        st = st.mask(L["_huy"].fillna(False).astype(bool) & ~st.eq("tra_hang"), "huy")
    if "_tra" in L.columns:
        ret = ret | L["_tra"].fillna(False).astype(bool)

    def any_by_order(mask):
        return mask.groupby(oid).transform("any").astype(bool)

    status = pd.Series(np.select([any_by_order(st.eq("huy")), any_by_order(ret), any_by_order(st.eq("dang_giao"))],
                                 ["huy", "tra_hang", "dang_giao"], "hoan_thanh"), index=L.index)
    base_ts = ts.where(~(neg | st.eq("tra_hang")))   # a return line may carry the return time
    first_ts = base_ts.groupby(oid).transform("min").fillna(ts.groupby(oid).transform("min"))
    tt = ts.dropna()
    has_time = len(tt) > 0 and (tt != tt.dt.normalize()).mean() > 0.3
    notes.append("order_ts from time" if has_time else "no time of day: order_ts NULL")
    order_ts = first_ts if has_time else pd.Series(pd.NaT, index=L.index, dtype="datetime64[ns]")

    V = pd.DataFrame({
        "order_id": oid, "order_date": first_ts.dt.normalize(), "order_ts": order_ts,
        "status": status, "is_cancelled": status.eq("huy"), "is_returned": status.eq("tra_hang"),
        "is_valid": ~status.isin(["huy", "tra_hang"]),
        "product_code": pd.Series(codes, index=L.index), "product_name": pd.Series(names, index=L.index),
        "product_group": pd.Series(group, index=L.index), "line_kind": kind, "unit": unit,
        "qty": q, "factor": factor, "qty_base": qbase, "base_unit": pd.Series(base, index=L.index),
        "unit_price": price, "amount": amount, "cost_unit": num("cost"),
        "customer_key": ckeys, "customer_code": ccode, "customer_name": cn,
        "customer_group": S("customer_group"), "payment": S("payment"), "channel": S("channel"),
        "staff": S("staff"),
    })

    # orders: order-level columns from the first line; revenue = item + service line amounts
    order_cols = ["order_date", "order_ts", "status", "is_cancelled", "is_returned", "is_valid", "customer_key",
                  "customer_code", "customer_name", "customer_group", "payment", "channel", "staff"]
    prod = V["line_kind"].isin(["item", "service"])
    O = V.groupby("order_id", sort=False)[order_cols].first()
    O["revenue"] = V.loc[prod].groupby("order_id")["amount"].sum().reindex(O.index).fillna(0.0)
    O["n_lines"] = V.loc[prod].groupby("order_id").size().reindex(O.index).fillna(0).astype(int)
    O = O.reset_index()
    info = {"notes": notes, "warnings": warns, "roles": I, "ev": lt["ev"]}
    return V, O, info


def _store(db, name, df):
    for c in df.columns:   # NaN / NA -> NULL in text columns
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    db.register("_views_tmp", df)
    try:
        db.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * REPLACE (CAST(order_date AS DATE) AS order_date) '
                   f'FROM _views_tmp')
    finally:
        db.unregister("_views_tmp")


def _describe(V, O, info):
    I, ev = info["roles"], info["ev"]
    out = []
    if V is not None:
        nv = int((V["is_valid"].fillna(False).astype(bool) & V["line_kind"].isin(["item", "service"])).sum())
        out += [f'VIEW "v_sales" ({nv} rows): item/service lines of VALID orders only; SUM(amount) = revenue as is',
                f'VIEW "v_sales_all" ({len(V)} rows): every line incl. cancelled/returned orders and adjust lines',
                "  columns (both): " + ", ".join(V.columns)]
    no = int(O["is_valid"].fillna(False).astype(bool).sum())
    out += [f'VIEW "v_orders" ({no} rows): VALID orders only (not cancelled, not returned); revenue = sum(amount) '
            f"of item/service lines (order amount when there are no item lines)",
            f'VIEW "v_orders_all" ({len(O)} rows): every order incl. cancelled/returned (is_cancelled, is_returned)',
            "  columns (both): " + ", ".join(O.columns)]
    roles = ", ".join(f"{r}<-{'header:' + c[4:] if c.startswith('__h_') else c}"
                      + (f" [{ev[c]}]" if c in ev else "") for r, c in I.items() if not r.startswith("__"))
    out.append("  roles: " + roles)
    out.append("  built from: " + "; ".join(info["notes"]))
    if info["warnings"]:
        out.append("VIEW WARNINGS: " + "; ".join(info["warnings"][:8]))
    return "\n".join(out)


def add_views(db, meta) -> str:
    """Create v_sales and v_orders in `db`; return their description for the LLM ("" if none)."""
    try:
        T = _load(db, meta)
        built = _build(T)
        if built is None:
            return ""
        V, O, info = built
        # safe by default: v_sales / v_orders hold only valid revenue lines / valid orders, so an
        # unfiltered SUM(amount) is already the rulebook revenue; *_all keep every line for cancel/return questions
        if V is not None:
            _store(db, "v_sales_all", V)
            _store(db, "v_sales", V[V["is_valid"].fillna(False).astype(bool) & V["line_kind"].isin(["item", "service"])].copy())
        _store(db, "v_orders_all", O)
        _store(db, "v_orders", O[O["is_valid"].fillna(False).astype(bool)].copy())
        return _describe(V, O, info)
    except Exception as e:   # a broken shop gets no views; the load must not fail
        print(f"views: skipped ({type(e).__name__}: {e})", file=sys.stderr)
        return ""
