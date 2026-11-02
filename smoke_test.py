"""Offline smoke tests for sec_extraction_langgraph.ipynb.

The notebook's pure-logic cells are tagged `logic`. `load_logic_cells` concatenates their source, the `ns`
fixture execs it into a fresh namespace, and `FakeLLM` stands in for the Azure OpenAI client, so nothing
here touches the network.   Run:  python -m pytest sec_extraction/smoke_test.py -v
"""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

NB_PATH = Path(__file__).with_name("sec_extraction_langgraph.ipynb")


def load_logic_cells(nb_path) -> str:
    """Source of every code cell tagged `logic`, joined in notebook order."""
    nb = nbformat.read(str(nb_path), as_version=4)
    return "\n\n".join(
        c.source for c in nb.cells
        if c.cell_type == "code" and "logic" in c.metadata.get("tags", [])
    )


# ---------------------------------------------------------------- fake LLM
def tool_response(name, args, usage=(10, 5)):
    """A chat.completions response whose message is a single tool call (args may be a dict or a raw string)."""
    raw = args if isinstance(args, str) else json.dumps(args)
    call = SimpleNamespace(id="call_1", type="function", function=SimpleNamespace(name=name, arguments=raw))
    return _response(SimpleNamespace(content=None, tool_calls=[call], refusal=None), usage)


def text_response(text, usage=(10, 5)):
    return _response(SimpleNamespace(content=text, tool_calls=None, refusal=None), usage)


def _response(message, usage):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]),
    )


class FakeLLM:
    """Stands in for AzureOpenAI. `handler(kwargs) -> response` decides each reply (thread-safe if pure);
    every call's kwargs are kept in `.calls`."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        with self._lock:
            self.calls.append(kw)
        return self.handler(kw)

    @classmethod
    def scripted(cls, responses):
        """Replies from a list in order (use only for sequential code)."""
        it = iter(responses)
        lock = threading.Lock()

        def handler(_kw):
            with lock:
                return next(it)
        return cls(handler)


@pytest.fixture(scope="session")
def logic_source():
    return load_logic_cells(NB_PATH)


@pytest.fixture()
def ns(logic_source):
    """A fresh namespace with every logic cell executed."""
    space = {"__name__": "nb"}
    exec(compile(logic_source, "<notebook logic cells>", "exec"), space)
    return space


# ------------------------------------------------------------------ tests
def test_logic_cells_load(tmp_path):
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("X = 1", metadata={"tags": ["logic"]}),
        nbformat.v4.new_code_cell("raise RuntimeError('network cell')"),
        nbformat.v4.new_markdown_cell("# title"),
    ]
    p = tmp_path / "nb.ipynb"
    nbformat.write(nb, str(p))
    assert load_logic_cells(p) == "X = 1"


def test_fake_llm_records_calls_and_usage():
    llm = FakeLLM.scripted([text_response("hi", usage=(3, 4))])
    r = llm.chat.completions.create(model="m", messages=[])
    assert r.choices[0].message.content == "hi"
    assert (r.usage.prompt_tokens, r.usage.completion_tokens) == (3, 4)
    assert llm.calls[0]["model"] == "m"


# ------------------------------------------------- Task 2: compile schema / questions
def _schema():
    return {
        "record_name": "financial_metric",
        "key": ["metric", "period"],
        "fields": [
            {"name": "metric", "type": "string", "description": "name", "required": True},
            {"name": "period", "type": "string", "description": "period", "required": True},
            {"name": "value", "type": "number", "description": "value", "required": True},
            {"name": "unit", "type": "enum", "values": ["USD", "%"], "description": "unit", "required": False},
        ],
    }


def _objects(node):
    """Yield every object-schema node in a JSON schema."""
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            yield node
        for v in node.values():
            yield from _objects(v)
    elif isinstance(node, list):
        for v in node:
            yield from _objects(v)


def test_compile_adds_page_and_quote(ns):
    c = ns["compile_schema"](_schema())
    props = c.json_schema["properties"]
    assert props["page"]["type"] == "integer"
    assert props["evidence_quote"]["type"] == "string"
    assert c.tool["function"]["name"] == "record_findings"
    assert c.tool["function"]["parameters"]["properties"]["records"]["items"] == c.json_schema


def test_strict_mode_rules(ns):
    c = ns["compile_schema"](_schema())
    for obj in _objects(c.tool["function"]["parameters"]):
        assert obj["additionalProperties"] is False
        assert sorted(obj["required"]) == sorted(obj["properties"])
    assert c.tool["function"]["strict"] is True
    assert "null" in c.json_schema["properties"]["unit"]["type"]      # optional -> nullable
    assert "null" not in c.json_schema["properties"]["value"]["type"]


def test_enum_field(ns):
    c = ns["compile_schema"](_schema())
    unit = c.json_schema["properties"]["unit"]
    assert unit["enum"] == ["USD", "%", None]
    ok = dict(metric="Revenue", period="FY24", value=1.0, unit="USD", page=1, evidence_quote="q")
    assert ns["validate_record"](c, ok).unit == "USD"
    with pytest.raises(Exception):
        ns["validate_record"](c, {**ok, "unit": "EUR"})


def test_key_must_be_declared_field(ns):
    bad = _schema()
    bad["key"] = ["metric", "nope"]
    with pytest.raises(ValueError):
        ns["compile_schema"](bad)


@pytest.mark.parametrize("mutate", [
    lambda s: s["fields"].append(dict(s["fields"][0])),                                   # duplicate name
    lambda s: s["fields"].append({"name": "x", "type": "blob", "description": "d", "required": True}),
    lambda s: s["fields"].append({"name": "y", "type": "enum", "description": "d", "required": True}),  # enum w/o values
    lambda s: s.__setitem__("fields", []),
])
def test_bad_schemas_rejected(ns, mutate):
    s = _schema()
    mutate(s)
    with pytest.raises(ValueError):
        ns["compile_schema"](s)


def test_hash_stable_and_changes_with_schema(ns):
    a, b = ns["compile_schema"](_schema()), ns["compile_schema"](_schema())
    assert a.hash == b.hash
    s = _schema()
    s["fields"][2]["description"] = "different"
    assert ns["compile_schema"](s).hash != a.hash


def test_numeric_string_rejected(ns):
    c = ns["compile_schema"](_schema())
    rec = dict(metric="Revenue", period="FY24", value="(1,234)", unit=None, page=1, evidence_quote="q")
    with pytest.raises(Exception):
        ns["validate_record"](c, rec)
    with pytest.raises(Exception):                       # strict: no silent coercion of "1234" either
        ns["validate_record"](c, {**rec, "value": "1234"})


def test_compile_questions(ns):
    qs = ns["compile_questions"](["  What is X? ", "", "What is Y?"])
    assert qs == [{"id": "q1", "text": "What is X?"}, {"id": "q2", "text": "What is Y?"}]
    with pytest.raises(ValueError):
        ns["compile_questions"]([" "])


# ------------------------------------------------- Task 3: CU result -> pages
def test_split_on_pagebreak_marker(ns):
    r = {"contents": [{"markdown": "alpha\n\n<!-- PageBreak -->\n\nbeta\n\n<!-- PageBreak -->\n\ngamma"}]}
    pages = ns["result_to_pages"](r)
    assert [p["n"] for p in pages] == [1, 2, 3]
    assert [p["text"] for p in pages] == ["alpha", "beta", "gamma"]


def test_fallback_to_page_spans(ns):
    r = {"contents": [{"markdown": "AAABBB", "pages": [
        {"pageNumber": 1, "spans": [{"offset": 0, "length": 3}]},
        {"pageNumber": 2, "spans": [{"offset": 3, "length": 3}]}]}]}
    assert [p["text"] for p in ns["result_to_pages"](r)] == ["AAA", "BBB"]


def test_single_page_no_markers(ns):
    pages = ns["result_to_pages"]({"contents": [{"markdown": "only page"}]})
    assert pages == [{"n": 1, "text": "only page"}]


def test_empty_result_gives_no_pages(ns):
    assert ns["result_to_pages"]({"contents": []}) == []


# ------------------------------------------------- Task 4: LLM helper, document map, chunking
def _pages(n, **special):
    return [{"n": i, "text": special.get(f"p{i}", f"body of page {i} total 1 unit")} for i in range(1, n + 1)]


def _ranges(chunks):
    return [(c.start, c.end) for c in chunks]


def test_chunk_boundaries_and_overlap(ns):
    chunks = ns["chunk_pages"](_pages(10), size=5, overlap=1)
    assert _ranges(chunks) == [(1, 5), (5, 9), (9, 10)]
    assert [c.id for c in chunks] == [0, 1, 2]
    assert [p["n"] for p in chunks[1].pages] == [5, 6, 7, 8, 9]


def test_split_on_headings_first(ns):
    pages = _pages(6, p4="# Item 7. Management's Discussion\ntext")
    assert _ranges(ns["chunk_pages"](pages, size=5, overlap=1)) == [(1, 3), (4, 6)]


def test_heading_found_within_first_lines(ns):
    pages = _pages(4, p3="\n\nAnnual Report\nItem 1A. Risk Factors\nblah")
    assert _ranges(ns["chunk_pages"](pages, size=5, overlap=1)) == [(1, 2), (3, 4)]


def test_chunk_carries_nearest_heading(ns):
    pages = _pages(8, p3="# Item 1A. Risk Factors\nx")
    chunks = ns["chunk_pages"](pages, size=3, overlap=0)
    assert chunks[0].heading == "" and chunks[1].heading == "Item 1A. Risk Factors"
    assert chunks[2].heading == "Item 1A. Risk Factors"           # inherited from earlier page


def test_table_not_split(ns):
    pages = _pages(8, p3="intro\n| a | b |\n| 1 | 2 |", p4="| 3 | 4 |\n| 5 | 6 |\nafter")
    chunks = ns["chunk_pages"](pages, size=3, overlap=0)
    assert any(c.start <= 3 and c.end >= 4 for c in chunks)       # pages 3 and 4 sit together


def test_single_page_doc(ns):
    chunks = ns["chunk_pages"](_pages(1), size=5, overlap=1)
    assert _ranges(chunks) == [(1, 1)]
    assert ns["chunk_pages"]([], size=5, overlap=1) == []


def test_token_budget_shrinks_but_always_progresses(ns):
    pages = [{"n": i, "text": "x" * 4000} for i in range(1, 7)]          # ~1000 tokens per page
    chunks = ns["chunk_pages"](pages, size=5, overlap=1, max_tokens=2500)
    covered = {p["n"] for c in chunks for p in c.pages}
    assert covered == set(range(1, 7))
    assert all(len(c.pages) <= 2 for c in chunks)
    one = ns["chunk_pages"]([{"n": 1, "text": "y" * 40000}], size=5, overlap=1, max_tokens=100)
    assert _ranges(one) == [(1, 1)]                                       # a single oversize page is still emitted


def test_render_contains_page_tags_and_ledger(ns):
    chunk = ns["chunk_pages"](_pages(2), size=5, overlap=0)[0]
    out = ns["render_chunk"](chunk, "MAP TEXT")
    assert out.index("<document_map>") < out.index("<pages>")
    assert '<page n="1">' in out and '<page n="2">' in out and "MAP TEXT" in out
    assert "<ledger>" not in out
    assert "<ledger>\nFOUND SO FAR\n</ledger>" in ns["render_chunk"](chunk, "MAP", ledger="FOUND SO FAR")


def test_call_llm_tracks_usage_and_effort(ns):
    ns["reset_run_state"]()
    llm = FakeLLM.scripted([text_response("ok", usage=(7, 3))])
    ns["call_llm"](llm, "m", "stage_x", [{"role": "user", "content": "hi"}])
    assert ns["USAGE"] == [{"stage": "stage_x", "in": 7, "out": 3}]
    assert llm.calls[0]["reasoning_effort"] == ns["REASONING_EFFORT"]
    assert ns["usage_by_stage"]()["stage_x"] == {"in": 7, "out": 3, "calls": 1}


# ---- JsonSchemaTools: gpt-6-astra on chat completions rejects function tools, so tool requests become response_format
_T1 = {"type": "function", "function": {"name": "rec", "description": "Report things.", "strict": True, "parameters": {
    "type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"], "additionalProperties": False}}}
_T2 = {"type": "function", "function": {"name": "other", "description": "Look up.", "parameters": {
    "type": "object", "properties": {"start": {"type": "integer"}}, "required": ["start"], "additionalProperties": False}}}


def test_default_task_prompt_matches_mode(ns):
    assert set(ns["TASK_PROMPTS"]) == {"structured", "verbose"}
    assert ns["TASK_PROMPT"] == ns["TASK_PROMPTS"][ns["MODE"]]
    assert "financial" not in ns["TASK_PROMPTS"]["verbose"].lower()      # must not contradict the questions


def test_adapter_forced_tool_becomes_json_schema(ns):
    inner = FakeLLM.scripted([text_response('{"x": 1}')])
    llm = ns["JsonSchemaTools"](inner)
    resp = llm.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}], tools=[_T1],
                                       tool_choice={"type": "function", "function": {"name": "rec"}},
                                       parallel_tool_calls=False, reasoning_effort="medium")
    kw = inner.calls[0]
    assert "tools" not in kw and "tool_choice" not in kw and "parallel_tool_calls" not in kw
    assert kw["reasoning_effort"] == "medium"
    js = kw["response_format"]["json_schema"]
    assert kw["response_format"]["type"] == "json_schema" and js["name"] == "rec" and js["strict"] is True
    assert js["schema"] == _T1["function"]["parameters"]
    tc = resp.choices[0].message.tool_calls[0]
    assert tc.function.name == "rec" and json.loads(tc.function.arguments) == {"x": 1}


def test_adapter_choice_between_tools_uses_envelope(ns):
    inner = FakeLLM.scripted([text_response(json.dumps({"tool": "other", "arguments_json": '{"start": 3}'}))])
    llm = ns["JsonSchemaTools"](inner)
    resp = llm.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}], tools=[_T1, _T2])
    kw = inner.calls[0]
    assert kw["response_format"]["json_schema"]["schema"]["properties"]["tool"]["enum"] == ["rec", "other"]
    note = [m for m in kw["messages"] if m["role"] == "system"][-1]["content"]
    assert "rec" in note and "other" in note and "Look up." in note
    tc = resp.choices[0].message.tool_calls[0]
    assert tc.function.name == "other" and json.loads(tc.function.arguments) == {"start": 3}


def test_adapter_flattens_tool_history_and_survives_bad_output(ns):
    inner = FakeLLM.scripted([text_response("not json at all")])
    llm = ns["JsonSchemaTools"](inner)
    history = [{"role": "user", "content": "q"},
               {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "other", "arguments": '{"start": 1}'}}]},
               {"role": "tool", "tool_call_id": "c1", "content": "PAGE TEXT"}]
    resp = llm.chat.completions.create(model="m", messages=history, tools=[_T1, _T2])
    sent = inner.calls[0]["messages"]
    assert all(m["role"] in ("system", "user", "assistant") and "tool_calls" not in m for m in sent)
    assert any("PAGE TEXT" in str(m["content"]) for m in sent)
    assert resp.choices[0].message.tool_calls is None          # unparseable reply -> callers see "no tool call"


def test_adapter_passes_plain_calls_through(ns):
    inner = FakeLLM.scripted([text_response("hello")])
    resp = ns["JsonSchemaTools"](inner).chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}], reasoning_effort="medium")
    assert "response_format" not in inner.calls[0] and resp.choices[0].message.content == "hello"


def test_document_map_uses_llm(ns):
    ns["reset_run_state"]()
    llm = FakeLLM.scripted([text_response("Form 10-K, ACME, FY2024")])
    pages = _pages(3, p2="# Item 1. Business\nstuff")
    assert ns["build_document_map"](pages, llm, "m") == "Form 10-K, ACME, FY2024"
    prompt = json.dumps(llm.calls[0]["messages"])
    assert "Item 1. Business" in prompt and "page 2" in prompt.lower()
    assert ns["usage_by_stage"]()["doc_map"]["calls"] == 1


# ------------------------------------------------- Task 5: map and reduce nodes
def _rec(metric="Revenue", period="FY24", value=100.0, unit="USD", page=2, quote="Revenue was 100", **extra):
    return dict(metric=metric, period=period, value=value, unit=unit, page=page, evidence_quote=quote, **extra)


def _chunk(ns, n=3):
    return ns["chunk_pages"](_pages(n), size=5, overlap=0)[0]


def _setup(ns):
    ns["reset_run_state"]()
    return ns["compile_schema"](_schema())


def test_prompt_user_task_comes_last(ns):
    p = ns["build_prompt"]("MY TASK TEXT", "structured")
    assert p.rstrip().endswith("</task>") and "MY TASK TEXT" in p
    assert "verbatim" in p.lower() and "record_findings" in p
    q = ns["build_prompt"]("T", "verbose", questions=[{"id": "q1", "text": "Why?"}])
    assert "q1" in q and "Why?" in q and q.rstrip().endswith("</task>")


def test_zero_records_ok(ns):
    c = _setup(ns)
    llm = FakeLLM.scripted([tool_response("record_findings", {"records": []})])
    assert ns["extract_structured"](_chunk(ns), "MAP", c, llm, "m", "task") == []
    assert ns["FAILURES"] == []


def test_extract_structured_prompt_layout_and_forced_tool(ns):
    c = _setup(ns)
    llm = FakeLLM.scripted([tool_response("record_findings", {"records": []})])
    ns["extract_structured"](_chunk(ns), "MAP", c, llm, "m", "MY TASK")
    kw = llm.calls[0]
    assert kw["tool_choice"]["function"]["name"] == "record_findings" and kw["tools"] == [c.tool]
    user = kw["messages"][-1]["content"]
    assert user.index("<pages>") < user.index("<task>") and user.rstrip().endswith("</task>") and "MY TASK" in user
    assert ns["usage_by_stage"]()["map"]["calls"] == 1


def test_extract_structured_validates_and_tags(ns):
    c = _setup(ns)
    bad = _rec(value="(1,234)", metric="Income")
    llm = FakeLLM.scripted([tool_response("record_findings", {"records": [_rec(), bad]})])
    out = ns["extract_structured"](_chunk(ns), "MAP", c, llm, "m", "task")
    assert out[0]["metric"] == "Revenue" and out[0]["_chunk"] == 0 and "_invalid" not in out[0]
    assert "_invalid" in out[1] and out[1]["metric"] == "Income"


@pytest.mark.parametrize("bad_reply", [
    lambda: tool_response("record_findings", "{not json"),
    lambda: text_response("I cannot help with that"),                       # refusal / no tool call
])
def test_invalid_reply_retried_then_logged(ns, bad_reply):
    c = _setup(ns)
    llm = FakeLLM.scripted([bad_reply(), bad_reply()])
    assert ns["extract_structured"](_chunk(ns), "MAP", c, llm, "m", "task") == []
    assert len(llm.calls) == 2 and len(ns["FAILURES"]) == 1 and ns["FAILURES"][0]["chunk"] == 0


def test_retry_recovers(ns):
    c = _setup(ns)
    llm = FakeLLM.scripted([tool_response("record_findings", "{bad"), tool_response("record_findings", {"records": [_rec()]})])
    out = ns["extract_structured"](_chunk(ns), "MAP", c, llm, "m", "task")
    assert len(out) == 1 and ns["FAILURES"] == []
    assert "unusable" in llm.calls[1]["messages"][-1]["content"].lower()


def test_ledger_reaches_the_prompt(ns):
    c = _setup(ns)
    llm = FakeLLM.scripted([tool_response("record_findings", {"records": []})])
    ns["extract_structured"](_chunk(ns), "MAP", c, llm, "m", "task", ledger="Revenue FY24 = 100")
    assert "<ledger>\nRevenue FY24 = 100\n</ledger>" in llm.calls[0]["messages"][-1]["content"]


def test_merge_dedupes_identical(ns):
    merged, conflicts = ns["merge_records"]([_rec(), _rec(page=3, quote="again"), _rec(metric="revenue ")], ["metric", "period"])
    assert len(merged) == 1 and conflicts == []


def test_merge_flags_conflict_same_key_different_value(ns):
    merged, conflicts = ns["merge_records"]([_rec(value=100.0), _rec(value=120.0, page=5)], ["metric", "period"])
    assert len(merged) == 2 and all(r["_conflict"] for r in merged)
    assert len(conflicts) == 1 and {r["value"] for r in conflicts[0]["records"]} == {100.0, 120.0}


def test_merge_prefers_record_with_evidence(ns):
    merged, conflicts = ns["merge_records"]([_rec(value=999.0, quote=""), _rec(value=100.0)], ["metric", "period"])
    assert len(merged) == 1 and merged[0]["value"] == 100.0 and conflicts == []


def test_merge_empty(ns):
    assert ns["merge_records"]([], ["metric"]) == ([], [])


QS = [{"id": "q1", "text": "A?"}, {"id": "q2", "text": "B?"}]


def test_evidence_empty_for_unaddressed_question(ns):
    ns["reset_run_state"]()
    reply = {"items": [{"question_id": "q1", "page": 2, "quote": "hello"}, {"question_id": "q9", "page": 1, "quote": "stray"}]}
    llm = FakeLLM.scripted([tool_response("report_evidence", reply)])
    ev = ns["extract_evidence"](_chunk(ns), "MAP", QS, llm, "m", "task")
    assert ev["q1"] == [{"page": 2, "quote": "hello", "_chunk": 0}] and ev["q2"] == [] and "q9" not in ev
    assert llm.calls[0]["tool_choice"]["function"]["name"] == "report_evidence"


def test_evidence_failure_logged(ns):
    ns["reset_run_state"]()
    llm = FakeLLM.scripted([text_response("no"), text_response("no")])
    assert ns["extract_evidence"](_chunk(ns), "MAP", QS, llm, "m", "task") == {"q1": [], "q2": []}
    assert len(ns["FAILURES"]) == 1


def test_answer_cites_only_pages_in_evidence(ns):
    ns["reset_run_state"]()
    llm = FakeLLM.scripted([tool_response("submit_answer", {"text": "The answer.", "citations": [3, 99]})])
    out = ns["answer_questions"]({"q1": [{"page": 3, "quote": "x"}], "q2": []}, QS, llm, "m", "task")
    assert out["q1"] == {"text": "The answer.", "citations": [3]}
    assert out["q2"]["citations"] == [] and "not addressed" in out["q2"]["text"].lower()
    assert len(llm.calls) == 1                                     # q2 needs no model call


def test_answer_hierarchical_for_large_evidence(ns):
    ns["reset_run_state"]()

    def handler(kw):
        return tool_response("submit_answer", {"text": "partial", "citations": [1]})
    llm = FakeLLM(handler)
    ev = {"q1": [{"page": 1, "quote": f"fact {i}"} for i in range(600)]}
    out = ns["answer_questions"](ev, QS[:1], llm, "m", "task", fanin=125)
    assert len(llm.calls) == 5 + 1 and out["q1"]["text"] == "partial"           # 5 partial answers, then one combine
    assert all(len(json.dumps(c["messages"])) < 40000 for c in llm.calls)


def test_answer_fits_a_few_hundred_quotes_in_one_call(ns):
    llm = FakeLLM(lambda kw: tool_response("submit_answer", {"text": "a", "citations": [1]}))
    ev = {"q1": [{"page": 1, "quote": f"fact {i}"} for i in range(200)]}
    ns["answer_questions"](ev, QS[:1], llm, "m", "task")
    assert len(llm.calls) == 1


def test_answers_run_in_parallel_and_report_progress(ns):
    import time as _t
    active, peak, lock = [0], [0], threading.Lock()

    def handler(kw):
        with lock:
            active[0] += 1; peak[0] = max(peak[0], active[0])
        _t.sleep(0.05)
        with lock:
            active[0] -= 1
        return tool_response("submit_answer", {"text": "a", "citations": [1]})
    msgs = []
    ev = {q["id"]: [{"page": 1, "quote": f"fact {i}"} for i in range(600)] for q in QS[:2]}
    out = ns["answer_questions"](ev, QS[:2], FakeLLM(handler), "m", "task", fanin=125, progress=msgs.append)
    assert peak[0] > 2 and set(out) == {q["id"] for q in QS[:2]}
    assert len([m for m in msgs if "answered" in m]) == 2 and any("600" in m for m in msgs)


# ------------------------------------------------- Task 6: verification and reviewer
PG = {1: "Intro text. Total revenue was $1,234 million in fiscal 2024.", 2: "Net income rose\nto $56 million, driven by growth.\n| Metric | FY24 |\n| Cash | 99 |",
      3: "Risk factors include competition."}


def test_exact_quote_passes(ns):
    assert ns["verify_quote"]("Total revenue was $1,234 million", PG, 1)


def test_ocr_noise_passes(ns):
    assert ns["verify_quote"]("Total revenue was $1,234 mi1lion in fisca1 2024", PG, 1)           # OCR l/1 confusion
    assert ns["verify_quote"]("net income rose to $56 million", PG, 2)                              # case + line break
    assert ns["verify_quote"]("| Cash | 99 |", PG, 2)                                              # table pipes


def test_html_table_markup_ignored(ns):
    pages = {1: "<table><tr><td>Total revenue</td><td>1,234</td></tr></table>"}
    assert ns["verify_quote"]("Total revenue 1,234", pages, 1)


def test_wrong_page_next_page_passes(ns):
    assert ns["verify_quote"]("Risk factors include competition", PG, 2)                          # model said 2, it is on 3
    assert not ns["verify_quote"]("Risk factors include competition", PG, 1)                      # two pages away


def test_quote_spanning_boundary_passes(ns):
    pages = {1: "revenue grew strongly and the board", 2: "approved a dividend of two dollars. Next."}
    assert ns["verify_quote"]("the board approved a dividend of two dollars", pages, 1)


def test_fabricated_quote_fails(ns):
    assert not ns["verify_quote"]("The company acquired Initech for $9 billion", PG, 1)
    assert not ns["verify_quote"]("", PG, 1)
    assert not ns["verify_quote"]("Total revenue", PG, 42)                                          # page does not exist


def test_verify_all_splits_and_routes_invalid(ns):
    items = [_rec(page=1, quote="Total revenue was $1,234 million"), _rec(page=1, quote="made up sentence about nothing"),
             {**_rec(page=1, quote="Total revenue was $1,234 million"), "_invalid": "bad value"}]
    ok, bad = ns["verify_all"](items, PG)
    assert len(ok) == 1 and len(bad) == 2
    ev = [{"page": 1, "quote": "Total revenue was"}, {"page": 1, "quote": "nonsense nonsense nonsense"}]
    ok, bad = ns["verify_all"](ev, PG, quote_key="quote")
    assert len(ok) == 1 and len(bad) == 1


def _review_llm(*replies):
    return FakeLLM.scripted(list(replies))


def test_reviewer_fixes_with_get_pages(ns):
    ns["reset_run_state"]()
    item = _rec(page=1, quote="invented quote about revenue", metric="Revenue")
    fixed = _rec(page=1, quote="Total revenue was $1,234 million", value=1234.0)
    llm = _review_llm(
        tool_response("get_pages", {"start": 1, "end": 2}),
        tool_response("submit_review", {"action": "fixed", "record_json": json.dumps(fixed)}))
    out = ns["review"]([item], PG, llm, "m")
    assert out[0]["action"] == "fixed" and out[0]["record"]["value"] == 1234.0
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs and "Total revenue was" in tool_msgs[0]["content"]


def test_reviewer_drops_when_unfixable(ns):
    ns["reset_run_state"]()
    llm = _review_llm(tool_response("submit_review", {"action": "dropped", "record_json": ""}))
    out = ns["review"]([_rec(quote="nope nope nope nope")], PG, llm, "m")
    assert out[0]["action"] == "dropped" and out[0]["record"] is None


def test_reviewer_fix_that_still_fails_verification_is_dropped(ns):
    ns["reset_run_state"]()
    bad_fix = _rec(page=1, quote="still invented text that is not there")
    llm = _review_llm(tool_response("submit_review", {"action": "fixed", "record_json": json.dumps(bad_fix)}))
    assert ns["review"]([_rec(quote="x y z w")], PG, llm, "m")[0]["action"] == "dropped"


def test_reviewer_stops_at_step_cap(ns):
    ns["reset_run_state"]()
    llm = FakeLLM(lambda kw: tool_response("get_pages", {"start": 1, "end": 1}))                  # never submits
    out = ns["review"]([_rec(quote="nope nope nope")], PG, llm, "m", max_steps=3)
    assert len(llm.calls) == 3 and out[0]["action"] == "dropped"


def test_reviewer_no_tool_call_drops(ns):
    ns["reset_run_state"]()
    llm = _review_llm(text_response("I think it is fine"))
    assert ns["review"]([_rec(quote="nope nope nope")], PG, llm, "m")[0]["action"] == "dropped"


# ------------------------------------------------- Task 7: graph, arms, comparison
import re as _re


def _first_page(kw):
    m = _re.search(r'<page n="(\d+)">', kw["messages"][-1]["content"])
    return int(m.group(1)) if m else None


def make_handler(*, bad_chunk_start=None, invalid_value=False, fix_value=1.0):
    """Deterministic fake model. Each map call reports one record/evidence on the first page of its chunk
    (quote = that page's text, so verification passes) plus one record shared by every chunk."""
    def handler(kw):
        tools = kw.get("tools") or []
        name = tools[0]["function"]["name"] if tools else None
        if name is None:
            return text_response("MAP: a fake 10-K")
        if name in ("record_findings", "report_evidence"):
            p = _first_page(kw)
            if p == bad_chunk_start:
                return text_response("sorry, no")
            quote = f"body of page {p} total 1 unit"
            if name == "record_findings":
                recs = [_rec(metric=f"M{p}", period="FY24", value=float(p), page=p, quote=quote),
                        _rec(metric="Shared", period="FY24", value=1.0, page=p, quote=quote)]
                if invalid_value:
                    recs.append(_rec(metric="Weird", period="FY24", value="(1,234)", page=p, quote=quote))
                return tool_response(name, {"records": recs})
            return tool_response(name, {"items": [{"question_id": "q1", "page": p, "quote": quote}]})
        if name == "submit_answer":
            return tool_response(name, {"text": "Answer.", "citations": [1, 5]})
        if name == "get_pages":                                               # reviewer
            return tool_response("submit_review", {"action": "fixed", "record_json": json.dumps(
                _rec(metric="Weird", period="FY24", value=fix_value, page=1, quote="body of page 1 total 1 unit"))})
        raise AssertionError(f"unexpected tool {name}")
    return handler


def _cfg(ns, llm, mode="structured", pages=None):
    compiled = ns["compile_schema"](_schema()) if mode == "structured" else None
    questions = ns["compile_questions"](["Why?", "How?"]) if mode == "verbose" else []
    return ns["RunConfig"](mode=mode, task_prompt="task", compiled=compiled, questions=questions, llm=llm, model="m",
                           pages=pages or _pages(8))


def test_main_arm_runs_end_to_end_structured(ns):
    llm = FakeLLM(make_handler())
    r = ns["run_arm"]("main", _cfg(ns, llm))
    assert sorted(x["metric"] for x in r.results) == ["M1", "M5", "Shared"]            # 2 chunks, shared record deduped
    assert r.verified_rate == 1.0 and r.conflicts == 0 and r.skipped == ""
    assert r.usage["map"]["calls"] == 2 and r.usage["doc_map"]["calls"] == 1 and r.tokens_in > 0 and r.seconds >= 0


def test_main_arm_runs_end_to_end_verbose(ns):
    llm = FakeLLM(make_handler())
    r = ns["run_arm"]("main", _cfg(ns, llm, mode="verbose"))
    assert r.results["answers"]["q1"]["text"] == "Answer." and r.results["answers"]["q1"]["citations"] == [1, 5]
    assert "not addressed" in r.results["answers"]["q2"]["text"].lower()
    assert sorted(e["page"] for e in r.results["evidence"]["q1"]) == [1, 5] and r.verified_rate == 1.0


def test_one_failed_chunk_does_not_stop_the_run(ns):
    llm = FakeLLM(make_handler(bad_chunk_start=5))
    r = ns["run_arm"]("main", _cfg(ns, llm))
    assert sorted(x["metric"] for x in r.results) == ["M1", "Shared"] and r.failures == 1


def test_invalid_record_goes_to_reviewer_and_is_fixed(ns):
    llm = FakeLLM(make_handler(invalid_value=True))
    r = ns["run_arm"]("main", _cfg(ns, llm))
    weird = [x for x in r.results if x["metric"] == "Weird"]
    assert len(weird) == 1 and weird[0]["value"] == 1.0
    assert r.usage["review"]["calls"] >= 1 and r.verified_rate < 1.0           # first-pass rate counts the failure


def test_ledger_passes_prior_state_to_next_chunk(ns):
    llm = FakeLLM(make_handler())
    ns["run_arm"]("ledger", _cfg(ns, llm))
    map_prompts = [c["messages"][-1]["content"] for c in llm.calls if c.get("tools") and c["tools"][0]["function"]["name"] == "record_findings"]
    assert len(map_prompts) == 2 and "<ledger>\n" not in map_prompts[0]      # scaffolding mentions <ledger> inline; the block starts a line
    assert "<ledger>\n" in map_prompts[1] and "M1" in map_prompts[1]


def test_baseline_single_call_within_budget(ns):
    llm = FakeLLM(make_handler())
    r = ns["run_arm"]("baseline", _cfg(ns, llm))
    assert r.usage["map"]["calls"] == 1 and "doc_map" not in r.usage
    assert any(x["metric"] == "M1" for x in r.results)


def test_baseline_skipped_over_budget(ns):
    ns["CONTEXT_BUDGET_TOKENS"] = 10
    llm = FakeLLM(make_handler())
    r = ns["run_arm"]("baseline", _cfg(ns, llm))
    assert "budget" in r.skipped.lower() and llm.calls == [] and r.results == []


def test_all_arms_report_same_config_hash_and_table(ns):
    llm = FakeLLM(make_handler())
    cfg = _cfg(ns, llm)
    runs = [ns["run_arm"](a, cfg) for a in ("baseline", "main", "ledger")]
    assert len({r.config_hash for r in runs}) == 1 and runs[0].config_hash == cfg.compiled.hash
    df = ns["comparison_df"](runs)
    assert list(df["arm"]) == ["baseline", "main", "ledger"]
    assert {"tokens_in", "tokens_out", "seconds", "items", "verified_rate", "conflicts", "config_hash"} <= set(df.columns)


def test_unknown_arm_rejected(ns):
    with pytest.raises(ValueError):
        ns["build_graph"]("nope", _cfg(ns, FakeLLM(make_handler())))


def test_every_code_cell_compiles_and_has_a_docstring():
    nb = nbformat.read(str(NB_PATH), as_version=4)
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        src = "\n".join(l for l in c.source.splitlines() if not l.startswith("%"))
        tree = compile(src, c.get("id", "?"), "exec", flags=__import__("ast").PyCF_ONLY_AST)
        assert __import__("ast").get_docstring(tree), f"cell {c.get('id')} has no top docstring"


# ------------------------------------------------- final-review fix pass (Important findings 1-4)
import httpx
import openai

_REQ = httpx.Request("POST", "https://example.invalid/chat/completions")


def _api_error(cls, status):
    return cls("boom", response=httpx.Response(status, request=_REQ), body=None)


def test_changed_digit_in_quote_fails_verification(ns):
    pages = {1: "Consolidated results. Total net sales were $ 391,035 million in fiscal 2024."}
    assert ns["verify_quote"]("Total net sales were $ 391,035 million", pages, 1)
    assert not ns["verify_quote"]("Total net sales were $ 391,935 million", pages, 1)      # one digit changed


def test_too_short_quote_fails(ns):
    assert not ns["verify_quote"]("net", {1: "net income was 5"}, 1)


def test_record_value_must_appear_in_quote(ns):
    pages = {1: "Total revenue was $1,234 million in fiscal 2024."}
    good = _rec(page=1, quote="Total revenue was $1,234 million", value=1234.0)
    wrong = _rec(page=1, quote="Total revenue was $1,234 million", value=999.0)
    label_only = _rec(page=1, quote="Total revenue was", value=1234.0)
    ok, bad = ns["verify_all"]([good, wrong, label_only], pages, numeric_fields=["value"])
    assert ok == [good] and len(bad) == 2


@pytest.mark.parametrize("value,quote,expected", [
    (-56.0, "Net loss was $ (56) million", True),                  # parentheses mean negative
    (1234000000.0, "Total revenue was $1,234 million", True),      # stated in millions, stored in dollars
    (1.234, "Total revenue was $1,234 million", True),             # stored in billions
    (12.5, "Gross margin was 12.5%", True),
])
def test_value_in_quote_allows_sign_and_scale(ns, value, quote, expected):
    pages = {1: quote}
    ok, bad = ns["verify_all"]([_rec(page=1, quote=quote, value=value)], pages, numeric_fields=["value"])
    assert bool(ok) is expected


def test_compiled_schema_lists_numeric_fields(ns):
    assert ns["compile_schema"](_schema()).numeric_fields == ["value"]


def test_prompt_tells_model_how_to_read_numbers(ns):
    assert "parenthes" in ns["build_prompt"]("t", "structured").lower()


def test_html_table_continuation_not_split(ns):
    pages = _pages(8, p3="intro\n<table><tr><td>a</td></tr></table>", p4="<table><tr><td>b</td></tr></table>\nafter")
    chunks = ns["chunk_pages"](pages, size=3, overlap=0)
    assert any(c.start <= 3 and c.end >= 4 for c in chunks)
    open_tbl = _pages(8, p3="intro\n<table><tr><td>a</td>", p4="<tr><td>b</td></tr></table>")
    assert any(c.start <= 3 and c.end >= 4 for c in ns["chunk_pages"](open_tbl, size=3, overlap=0))


def test_bad_request_is_fatal_not_swallowed(ns):
    c = _setup(ns)
    def boom(kw):
        raise _api_error(openai.BadRequestError, 400)
    llm = FakeLLM(boom)
    with pytest.raises(openai.BadRequestError):
        ns["extract_structured"](_chunk(ns), "MAP", c, llm, "m", "task")
    assert len(llm.calls) == 1 and ns["FAILURES"] == []


def test_rate_limit_backs_off_then_succeeds(ns):
    c = _setup(ns)
    ns["_BACKOFF"] = (0, 0, 0)
    state = {"n": 0}
    def flaky(kw):
        state["n"] += 1
        if state["n"] == 1:
            raise _api_error(openai.RateLimitError, 429)
        return tool_response("record_findings", {"records": []})
    llm = FakeLLM(flaky)
    assert ns["extract_structured"](_chunk(ns), "MAP", c, llm, "m", "task") == [] and len(llm.calls) == 2
    assert ns["FAILURES"] == []


def test_run_arm_aborts_when_every_map_call_fails(ns):
    def handler(kw):
        return text_response("MAP") if not kw.get("tools") else text_response("no tool call, sorry")
    with pytest.raises(RuntimeError, match="every map call failed"):
        ns["run_arm"]("main", _cfg(ns, FakeLLM(handler)))


def test_review_survives_api_error(ns):
    ns["reset_run_state"]()
    def boom(kw):
        raise RuntimeError("503 upstream")
    out = ns["review"]([_rec(quote="nope nope nope")], PG, FakeLLM(boom), "m")
    assert out[0]["action"] == "dropped" and "failed" in out[0]["reason"]


# ------------------------------------------------- speed and feedback on long runs
def test_map_calls_use_map_reasoning_effort(ns):
    ns["reset_run_state"]()
    llm = FakeLLM.scripted([text_response("a"), text_response("b")])
    ns["call_llm"](llm, "m", "map", [{"role": "user", "content": "x"}])
    ns["call_llm"](llm, "m", "answer", [{"role": "user", "content": "x"}])
    assert llm.calls[0]["reasoning_effort"] == ns["MAP_REASONING_EFFORT"] == "low"
    assert llm.calls[1]["reasoning_effort"] == ns["REASONING_EFFORT"]


def test_run_arm_reports_progress(ns):
    msgs = []
    cfg = _cfg(ns, FakeLLM(make_handler()))
    cfg.on_progress = msgs.append
    ns["run_arm"]("main", cfg)
    text = "\n".join(msgs)
    assert "2 chunks" in text and "map 1/2" in text and "map 2/2" in text and "verify" in text


def test_ledger_arm_reports_progress_and_handles_more_chunks_than_langgraph_default_limit(ns):
    msgs = []
    pages = _pages(150)                          # ~38 chunks: more steps than LangGraph's default recursion limit of 25
    cfg = _cfg(ns, FakeLLM(make_handler()), pages=pages)
    cfg.on_progress = msgs.append
    ns["run_arm"]("ledger", cfg)
    assert any(m.startswith("  map 1/") for m in msgs) and any("/38" in m or "/37" in m or "/39" in m for m in msgs)


# ------------------------------------------------- token counts in the outputs
def test_usage_table_breaks_tokens_down_by_arm_and_stage(ns):
    r = ns["run_arm"]("main", _cfg(ns, FakeLLM(make_handler())))
    t = ns["usage_table"]([r])
    assert list(t.columns) == ["arm", "stage", "calls", "tokens_in", "tokens_out", "tokens_total"]
    assert {"doc_map", "map"} <= set(t[t.arm == "main"].stage)
    total = t[t.stage == "TOTAL"].iloc[0]
    assert total.tokens_in == r.tokens_in and total.tokens_out == r.tokens_out and total.tokens_total == r.tokens_in + r.tokens_out
    assert total.calls == sum(u["calls"] for u in r.usage.values())


def test_comparison_df_has_total_tokens(ns):
    r = ns["run_arm"]("main", _cfg(ns, FakeLLM(make_handler())))
    row = ns["comparison_df"]([r]).iloc[0]
    assert row.tokens_total == row.tokens_in + row.tokens_out > 0


# ------------------------------------------------- local PDF reader (no Content Understanding)
def _make_pdf(path, page_texts):
    """A minimal valid PDF with a real text layer (Helvetica), one string per page ('' -> a page with no text)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None]
    kids, n = [], 3
    font_obj = 3 + 2 * len(page_texts)
    for t in page_texts:
        esc = t.replace(chr(92), chr(92) * 2).replace("(", chr(92) + "(").replace(")", chr(92) + ")")
        stream = f"BT /F1 12 Tf 72 720 Td ({esc}) Tj ET" if t else ""
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {n + 1} 0 R /Resources << /Font << /F1 {font_obj} 0 R >> >> >>")
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        kids.append(f"{n} 0 R"); n += 2
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(page_texts)} >>"
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out)); out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode() + "".join(f"{o:010d} 00000 n \n" for o in offs).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    Path(path).write_bytes(out)


def test_local_reader_returns_one_numbered_text_per_page(ns, tmp_path):
    _make_pdf(tmp_path / "f.pdf", ["Item 1. Business. Revenue was 41 million " * 3, "Item 1A. Risk Factors. Competition is intense " * 3, "Item 7. Results " * 6])
    pages = ns["read_pdf_local"](tmp_path / "f.pdf")
    assert [p["n"] for p in pages] == [1, 2, 3]
    assert "Revenue was 41 million" in pages[0]["text"] and "Risk Factors" in pages[1]["text"]
    assert all(set(p) == {"n", "text"} for p in pages)


def test_local_reader_pages_feed_the_rest_of_the_flow(ns, tmp_path):
    _make_pdf(tmp_path / "f.pdf", [f"Item {i}. Section. Revenue was {i} million " * 4 for i in range(1, 7)])
    pages = ns["read_pdf_local"](tmp_path / "f.pdf")
    chunks = ns["chunk_pages"](pages)
    assert chunks and chunks[0].pages[0]["n"] == 1
    assert ns["verify_quote"]("Revenue was 3 million", {p["n"]: p["text"] for p in pages}, 3)


def test_local_reader_explains_a_pdf_without_a_text_layer(ns, tmp_path):
    _make_pdf(tmp_path / "scan.pdf", ["", "", ""])
    with pytest.raises(ValueError, match="text layer|scanned|OCR"):
        ns["read_pdf_local"](tmp_path / "scan.pdf")


def test_content_understanding_flag_exists(ns):
    assert isinstance(ns["USE_CONTENT_UNDERSTANDING"], bool)


# ------------------------------------------------- ledger arm in waves (faster than strictly one-by-one)
def test_ledger_wave_runs_a_wave_in_parallel_and_later_waves_see_the_ledger(ns):
    import time as _t
    inner = make_handler()
    active, peak, lock = [0], [0], threading.Lock()

    def handler(kw):
        with lock:
            active[0] += 1; peak[0] = max(peak[0], active[0])
        _t.sleep(0.03)
        with lock:
            active[0] -= 1
        return inner(kw)
    llm = FakeLLM(handler)
    cfg = _cfg(ns, llm, pages=_pages(40))
    cfg.ledger_wave = 4
    ns["run_arm"]("ledger", cfg)
    prompts = [c["messages"][-1]["content"] for c in llm.calls if c.get("tools") and c["tools"][0]["function"]["name"] == "record_findings"]
    assert len(prompts) > 8 and peak[0] >= 3
    assert all("<ledger>\n" not in p for p in prompts[:4])                  # first wave: nothing known yet
    assert all("<ledger>\n" in p for p in prompts[4:])                      # later waves: told what earlier waves found


def test_ledger_wave_default_is_strictly_sequential_and_config_sets_a_faster_default(ns):
    assert ns["RunConfig"].__dataclass_fields__["ledger_wave"].default == 1
    assert ns["LEDGER_WAVE"] >= 2


def test_ledger_wave_reports_chunks_done_not_waves(ns):
    msgs = []
    cfg = _cfg(ns, FakeLLM(make_handler()), pages=_pages(40))
    cfg.ledger_wave = 4; cfg.on_progress = msgs.append
    ns["run_arm"]("ledger", cfg)
    last = [m for m in msgs if m.startswith("  map ")][-1]
    n = int(last.split("/")[1].split()[0])
    assert last.startswith(f"  map {n}/{n}")


# ------------------------------------------------- reviewer runs items in parallel
def test_review_handles_failed_items_in_parallel_and_keeps_order(ns):
    import time as _t
    active, peak, lock = [0], [0], threading.Lock()

    def handler(kw):
        with lock:
            active[0] += 1; peak[0] = max(peak[0], active[0])
        _t.sleep(0.05)
        with lock:
            active[0] -= 1
        return tool_response("submit_review", {"action": "dropped", "record_json": ""})
    llm = FakeLLM(handler)
    items = [{"metric": f"M{i}", "page": 1, "evidence_quote": "nope"} for i in range(12)]
    out = ns["review"](items, {1: "page one"}, llm, "m")
    assert [r["item"]["metric"] for r in out] == [f"M{i}" for i in range(12)]       # order preserved
    assert all(r["action"] == "dropped" for r in out) and peak[0] >= 4


# ------------------------------------------------- USE_REVIEW switch
def _review_calls(llm):
    return [c for c in llm.calls if c.get("tools") and any(t["function"]["name"] == "submit_review" for t in c["tools"])]


def test_use_review_flag_exists_and_defaults_to_on(ns):
    assert isinstance(ns["USE_REVIEW"], bool) and ns["USE_REVIEW"] is True
    assert ns["RunConfig"].__dataclass_fields__["use_review"].default is True


def test_use_review_false_skips_reviewer_calls_and_drops_failed_items(ns):
    msgs = []
    llm_on = FakeLLM(make_handler(invalid_value=True))
    cfg_on = _cfg(ns, llm_on, pages=_pages(12))
    on = ns["run_arm"]("main", cfg_on)
    assert _review_calls(llm_on)                         # precondition: this fake data does reach the reviewer

    llm_off = FakeLLM(make_handler(invalid_value=True))
    cfg_off = _cfg(ns, llm_off, pages=_pages(12))
    cfg_off.use_review = False; cfg_off.on_progress = msgs.append
    off = ns["run_arm"]("main", cfg_off)
    assert not _review_calls(llm_off)
    assert any("review skipped" in m for m in msgs)
    assert off.usage.get("review", {}).get("calls", 0) == 0
    assert off.n_items <= on.n_items
