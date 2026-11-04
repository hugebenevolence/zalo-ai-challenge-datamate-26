"""The notebooks' library cells (01_data, 02_agent) under test: normalization, data layer, voting,
answer handling and the agent end to end with a scripted model (no GPU)."""
import json
import os

import pandas as pd
import pytest

from datamate.nblib import run_lib_cells

from .conftest import REPO, TRAIN


@pytest.fixture(scope="session")
def nb():
    os.chdir(REPO / "notebooks")
    ns = {}
    run_lib_cells("02_agent.ipynb", ns)   # also loads 01_data's library cells
    return ns


# ---------------------------------------------------------------- question normalization

@pytest.mark.parametrize("raw, want", [
    ("dthu hqua cua cua hang la bn z", "doanh thu hôm qua cua cua hang la bao nhiêu vậy"),
    ("Dthu th1/2026 đc bnhiu z?", "doanh thu tháng 1/2026 được bao nhiêu vậy?"),
    ("T10/2025 dthu hơn hay kém", "tháng 10/2025 doanh thu hơn hay kém"),
    ("3/26 thợ mô làm nhiều nhất rứa?", "3/26 thợ nào làm nhiều nhất vậy?"),
    ("bình quân mỗi o đờ", "bình quân mỗi đơn"),
])
def test_expands(nb, raw, want):
    assert nb["expand_abbreviations"](raw) == want


@pytest.mark.parametrize("text", ["check giùm", "hơn hay kém", "biển số 59-K1", "giá 55k", "mua nhiu đơn nhất",
                                  "Ống PVC Bình Minh D50 cây 4m", "Topping Thạch cà phê", "Size XL"])
def test_leaves_alone(nb, text):
    assert nb["expand_abbreviations"](text) == text


# ---------------------------------------------------------------- data layer

def test_normalization_on_a_hand_kept_ledger(nb):
    biz = nb["load_business_cached"]("public_test", "dev_004")
    key = "ALL SoBanHang.xlsx::T*"
    assert biz.combined[key][0] == "SoBanHang.xlsx::T01-2025" and len(biz.combined[key]) == 18
    df = biz.frames[key]
    assert df["Thời gian__date"].dtype.kind == "M" and df["Thời gian__date"].notna().all()  # '1/1' + sheet year
    assert df["Amount__num"].notna().mean() > 0.99                                           # '5tr520', '90k'
    assert biz.filled[key] == "Ma don" and df["Ma don"].notna().all()
    blank_lines = df.groupby("Ma don")["Khách hàng (tên)"].apply(lambda s: s.isna().any() and s.notna().any())
    assert not blank_lines.any()                                                              # filled down


def test_description_points_to_parsed_columns(nb):
    biz = nb["load_business_cached"]("public_test", "dev_004")
    d = nb["describe_business"](biz)
    assert "use 'Thời gian__date' (datetime64" in d and "'ALL SoBanHang.xlsx::T*'" in d
    assert d.count("### tables['SoBanHang.xlsx::T0") == 0            # parts described once, via the ALL view


def test_parse_plan_never_reads_plain_ints_as_dates(nb):
    df = pd.DataFrame({"tien": [45000, 46000, 45500] * 10, "ngay": ["01/04/2025", "02/04/2025", "1 thg 5 2025"] * 10})
    plan = nb["plan_parsing"](df.astype(object), "x")
    assert plan["tien"][0] == "number" and plan["ngay"][0] == "date"     # ints get a __num copy, never a date


def test_number_parse_skips_phrases_and_phone_numbers(nb):
    df = pd.DataFrame({"addr": ["93 Nguyễn Thị Minh Khai", "12 Lê Lợi", "5 Tỉnh Lộ 10"] * 10,
                       "sdt": ["0914430525", "0980004652", "0903123456"] * 10, "sl": ["2 đĩa", "1 đĩa", "3 lon"] * 10})
    plan = nb["plan_parsing"](df.astype(object), "x")
    assert "addr" not in plan and "sdt" not in plan and plan["sl"][0] == "number"


def test_line_number_is_not_an_order_id(nb):
    df = pd.DataFrame({"STT": [1, 2, 1], "Mã phiếu nhập": ["PN1", "PN1", "PN2"]})
    assert nb["order_id_column"](df, {"STT": "số thứ tự dòng trong phiếu nhập"}) != "STT"
    assert nb["order_id_column"](df, {"STT": "số thứ tự đơn (hoá đơn) trong sổ"}) == "STT"


def test_independent_records_are_not_grouped(nb):
    df = pd.DataFrame({"Mã KH": ["K1", "K2", "K3", "K4", "K5"],
                       "Sinh nhật": ["01/02/1990", None, "03/04/1985", None, None],
                       "Email": ["a@x", None, "c@x", None, None]})
    assert nb["continuation_rows"](df, "khach_hang.csv")[0] is None


def test_order_ids_written_as_one_word(nb):
    for name in ["MaHoaDon", "ma_don", "SoHD", "OrderID"]:
        assert nb["order_id_column"](pd.DataFrame({name: ["A"], "SL": [1]}), {}) == name
    assert nb["order_id_column"](pd.DataFrame({"MaHangHoa": ["A"]}), {}) is None


def test_duplicate_lines_count_once_within_an_order_only(nb):
    df = pd.DataFrame({"Ma don": ["D1", "D1", "D1", "D2"], "Hang": ["Bia", "Bia", "Coca", "Bia"], "SL": [2, 2, 1, 2]})
    out, n = nb["dedupe_order_table"](df, {"Ma don": "mã đơn"})
    assert n == 1 and list(out["Ma don"]) == ["D1", "D1", "D2"]       # D2's identical line is another sale
    ledger = pd.DataFrame({"Ngay": ["1/4", "1/4"], "Tien": [5, 5]})
    assert nb["dedupe_order_table"](ledger, {})[1] == 0               # not kept by order: both lines count


def test_header_repeated_inside_a_table_is_dropped(nb):
    df = pd.DataFrame({"Mã đơn": ["D1", "Mã đơn", "D2"], "Tên hàng": ["Bia", "Tên hàng", "Coca"], "SL": [1, "SL", 2]})
    assert list(nb["drop_header_repeats"](df)["Mã đơn"]) == ["D1", "D2"]


def test_duplicate_order_lines_removed_on_load(nb):
    biz = nb["load_business_cached"]("train", "biz_123")             # sqlite POS, ids like MaHoaDon
    d = biz.frames["chi_tiet_don"]
    assert not d.duplicated().any() and len(d) == 64149 - 1208


# ---------------------------------------------------------------- answers and voting

N = lambda v: {"type": "number", "answer": v}


def test_vote_majority_support_counts_failures(nb):
    v = nb["vote"]([N(10), N(10), N(12), None, N(10.0)], 5)
    assert v["answer"] == N(10) and v["support"] == 0.6 and [c["size"] for c in v["clusters"]] == [3, 1]


def test_vote_clusters_like_the_scorer(nb):
    a, b = {"type": "string", "answer": "Sting dâu 330ml"}, {"type": "string", "answer": "STING DÂU 330ML"}
    assert nb["answer_key"](a) == nb["answer_key"](b)
    t1 = {"type": "table", "answer": {"columns": ["a", "b"], "rows": [["X", 1], ["Y", 2]]}}
    t2 = {"type": "table", "answer": {"columns": ["a", "b"], "rows": [["y", 2], ["x", 1]]}}
    assert nb["answer_key"](t1) == nb["answer_key"](t2)
    assert nb["answer_key"](N(5)) != nb["answer_key"]({"type": "percent", "answer": 5})


def test_decide(nb):
    v = nb["vote"]([N(1), N(2), None], 3)
    assert nb["decide"](v, 0.5) == {"type": "abstain"} and nb["decide"](v, 0.3) == N(1)
    u = {"type": "unanswerable", "reason": "x"}
    assert nb["decide"](nb["vote"]([u, u, N(3)], 3), 0.6) == u


@pytest.mark.parametrize("raw, want", [
    ({"type": "number", "answer": 12.0}, N(12)),
    ({"type": "percent", "answer": -12.4567}, {"type": "percent", "answer": -12.46}),
    ({"type": "list", "answer": [3.0, " Bạc xỉu "]}, {"type": "list", "answer": [3, "Bạc xỉu"]}),
    ({"type": "unanswerable"}, {"type": "unanswerable", "reason": ""}),
])
def test_normalize_answer(nb, raw, want):
    assert nb["normalize_answer"](raw) == want


@pytest.mark.parametrize("raw", [N("48250000"), {"type": "date", "answer": "15/08/2026"},
                                 {"type": "boolean", "answer": 1}, {"answer": 3}, {"type": "foo", "answer": 1}])
def test_normalize_answer_rejects(nb, raw):
    with pytest.raises(ValueError):
        nb["normalize_answer"](raw)


def test_percent_guard(nb):
    assert nb["postprocess"]("tỉ lệ đơn huỷ là bao nhiêu %", N(12.5)) == {"type": "percent", "answer": 12.5}
    assert nb["postprocess"]("doanh thu tháng 5", N(12)) == N(12)
    day = {"type": "string", "answer": "2025-11-15"}
    assert nb["postprocess"]("ngày nào doanh thu cao nhất", day) == {"type": "date", "answer": "2025-11-15"}
    assert nb["postprocess"]("ai mua nhiều nhất", {"type": "string", "answer": "Anh Ba"})["type"] == "string"


def test_walk_in_customer_is_never_a_ranked_customer(nb):
    S = lambda v: {"type": "string", "answer": v}
    assert nb["postprocess"]("khách nào mua nhiều đơn nhất?", S("Khách lẻ")) is None
    assert nb["postprocess"]("ai mua nhiều tiền nhất", S("kl")) is None
    assert nb["postprocess"]("3 khách chi nhiều nhất", {"type": "list", "answer": ["Chị Lan", "khách lẻ"]}) is None
    assert nb["postprocess"]("khách nào mua nhiều đơn nhất?", S("Chị Lan")) == S("Chị Lan")
    assert nb["postprocess"]("nhóm khách nào mua nhiều nhất", S("lẻ")) == S("lẻ")   # 'lẻ' can be a group name
    assert nb["postprocess"]("món nào bán chạy nhất", S("Bánh lẻ")) == S("Bánh lẻ")


def test_prompt_v2_is_unchanged_and_v3_differs(nb):
    assert "Parse every raw cell with the helpers" in nb["SYSTEM_V2"]
    assert "__date" in nb["SYSTEM_V3"] and "Parse every raw cell with the helpers" not in nb["SYSTEM_V3"]


# ---------------------------------------------------------------- agent end to end

GOOD = "```python\nANSWER = {'type': 'number', 'answer': 42.0}\n```"
OTHER = "```python\nANSWER = {'type': 'number', 'answer': 7}\n```"
BROKEN = "```python\n1/0\n```"
QS = [{"id": "q1", "business_id": "biz_110", "question": "dthu hqua la bn", "as_of_date": "2026-07-03"},
      {"id": "q2", "business_id": "biz_110", "question": "tỉ lệ đơn huỷ tháng 6/2026 là bao nhiêu %",
       "as_of_date": "2026-07-03"}]


def _agent(nb, llm, **cfg):
    return nb["Agent"](llm, nb["AgentConfig"](**cfg), (TRAIN / "rulebook.md").read_text(encoding="utf-8"),
                       TRAIN / "calendar.csv")


def _answer(nb, agent, qs):
    biz = nb["load_business_cached"]("train", "biz_110")
    return agent.answer_business(biz, nb["sandbox_tables"]("train", biz), qs)


def test_vote_threshold_and_percent_guard(nb):
    llm = nb["ScriptedChat"]([[GOOD, GOOD, OTHER]])
    run = _answer(nb, _agent(nb, llm, k=3, tau=0.6), QS)
    assert run.rows == [{"id": "q1", "type": "number", "answer": 42}, {"id": "q2", "type": "percent", "answer": 42.0}]
    assert llm.calls[0]["n"] == 3 and llm.calls[0]["temperature"] == 0.7
    assert run.logs[0]["support"] == pytest.approx(2 / 3) and len(run.logs[0]["samples"]) == 3


def test_abstains_below_threshold(nb):
    run = _answer(nb, _agent(nb, nb["ScriptedChat"]([[GOOD, OTHER, BROKEN]], [BROKEN]), k=3, tau=0.6), QS[:1])
    assert run.rows == [{"id": "q1", "type": "abstain"}]


def test_every_failed_sample_is_repaired_once(nb):
    llm = nb["ScriptedChat"]([[BROKEN, BROKEN]], [GOOD])
    run = _answer(nb, _agent(nb, llm, k=2, tau=0.5), QS[:1])
    assert len(llm.calls) == 2 and llm.calls[1]["temperature"] == 0.0 and len(llm.calls[1]["conversations"]) == 2
    assert "ZeroDivisionError" in llm.calls[1]["conversations"][0][-1]["content"]
    assert run.rows == [{"id": "q1", "type": "number", "answer": 42}]
    assert [s.get("repair_of") for s in run.logs[0]["samples"]] == [None, None, 0, 1]


def test_a_repaired_sample_votes_in_place_of_the_failed_one(nb):
    llm = nb["ScriptedChat"]([[GOOD, BROKEN, OTHER]], [GOOD])
    run = _answer(nb, _agent(nb, llm, k=3, tau=0.6), QS[:1])
    assert len(llm.calls[1]["conversations"]) == 1                      # only the broken sample is retried
    assert run.rows == [{"id": "q1", "type": "number", "answer": 42}] and run.logs[0]["support"] == pytest.approx(2 / 3)


def test_prompt_wiring(nb):
    llm = nb["ScriptedChat"]([GOOD])
    _answer(nb, _agent(nb, llm, k=1, prompt="v3"), QS[:1])
    system, user = (m["content"] for m in llm.calls[0]["conversations"][0])
    assert "# RULEBOOK" in system and "README.md of biz_110" in system and "{rulebook}" not in system
    assert "parse_number(x)" in system and "doanh thu hôm qua" in user and llm.calls[0]["temperature"] == 0.0


def test_run_split_writes_resumes_and_shards(nb, tmp_path):
    ids = sorted(json.loads(line)["id"] for line in (TRAIN / "questions.jsonl").open()
                 if '"biz_110"' in line or '"biz_114"' in line)[:4]
    cfg = nb["AgentConfig"](k=1)
    for i in range(2):
        nb["run_split"]("train", tmp_path, cfg, set(ids), nb["ScriptedChat"]([GOOD]), shard=f"{i}/2")
    lines = (tmp_path / "submission.jsonl").read_text().splitlines()
    assert sorted(json.loads(x)["id"] for x in lines) == ids and len(list((tmp_path / "logs").glob("*.json"))) == 4
    nb["run_split"]("train", tmp_path, cfg, set(ids), nb["ScriptedChat"]([BROKEN]), shard="0/2")  # resumes: no new work
    assert all(json.loads(x)["answer"] == 42 for x in (tmp_path / "submission.jsonl").read_text().splitlines())


def test_assign_shops_balances_questions_and_is_stable(nb):
    counts = {"a": 120, "b": 100, "c": 60, "d": 40, "e": 20}
    parts = nb["assign_shops"](counts, 2)
    assert sorted(sum(parts, [])) == sorted(counts) and parts == nb["assign_shops"](counts, 2)
    assert abs(sum(counts[b] for b in parts[0]) - sum(counts[b] for b in parts[1])) <= 20
    assert nb["assign_shops"](counts, 1) == [sorted(counts)]


def test_resume_drops_torn_lines_repeats_and_rows_without_a_log(nb, tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    for q in ("q1", "q2"):
        (logs / f"{q}.json").write_text("{}")
    sub = tmp_path / "submission.jsonl"
    sub.write_text('{"id": "q1", "type": "abstain"}\n{"id": "q1", "type": "abstain"}\n'
                   '{"id": "q3", "type": "abstain"}\n{"id": "q2", "ty')
    assert nb["load_done"](sub, logs) == {"q1"}
    assert sub.read_text() == '{"id": "q1", "type": "abstain"}\n'


def test_run_split_logs_runs_and_progress(nb, tmp_path):
    ids = sorted(json.loads(line)["id"] for line in (TRAIN / "questions.jsonl").open() if '"biz_110"' in line)[:2]
    nb["run_split"]("train", tmp_path, nb["AgentConfig"](k=1), set(ids), nb["ScriptedChat"]([GOOD]))
    events = [json.loads(x)["event"] for x in (tmp_path / "runs.jsonl").read_text().splitlines()]
    progress = [json.loads(x) for x in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert events == ["start", "end"] and progress[0]["shop"] == "biz_110" and progress[0]["questions"] == 2
    start = json.loads((tmp_path / "runs.jsonl").read_text().splitlines()[0])
    assert start["git_commit"] and start["config"]["k"] == 1 and "versions" in start


class _MetaChat:
    """A scripted model that also reports token counts and finish reasons, like VLLMChatModel."""

    def __init__(self, nb, reply):
        self.inner = nb["ScriptedChat"]([reply])

    def chat(self, conversations, n=1, temperature=0.0, thinking=0):
        out = self.inner.chat(conversations, n, temperature, thinking)
        self.last_meta = [[(123, "stop") for _ in per] for per in out]
        return out


def test_sample_logs_carry_tokens_and_finish_reason(nb):
    run = _answer(nb, _agent(nb, _MetaChat(nb, GOOD), k=2, tau=0.5), QS[:1])
    assert [(s["gen_tokens"], s["finish"]) for s in run.logs[0]["samples"]] == [(123, "stop")] * 2
    assert run.gen_tokens == 246 and run.truncated == 0 and len(run.logs[0]["system_sha"]) == 16


@pytest.mark.parametrize("name", ["01_data.ipynb", "02_agent.ipynb", "03_evaluate.ipynb"])
def test_committed_notebooks_carry_no_outputs(name):
    """Outputs hold competition data (customer names, questions): regulations §4.3. Clear them before committing."""
    import nbformat
    cells = nbformat.read(REPO / "notebooks" / name, as_version=4).cells
    assert not any(c.get("outputs") or c.get("execution_count") for c in cells if c.cell_type == "code")
