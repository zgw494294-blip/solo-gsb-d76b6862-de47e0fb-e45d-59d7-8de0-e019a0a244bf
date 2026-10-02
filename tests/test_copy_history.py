"""工作副本历史回看：版本快照记录、同事务提交、baseline 补记与 HTTP 接口测试。"""

import importlib
import sqlite3
import threading

import pytest

from app.copies import (
    CopyNotFound,
    CopyStillHeldError,
    CopyStore,
    GroupHeldError,
    ModifyConflictError,
    RevisionMismatchError,
    RevisionNotFound,
)

TEXT = "x" * 30


def ann(id, start, end, label="L"):
    return {"id": id, "start": start, "end": end, "label": label}


@pytest.fixture()
def store(tmp_path):
    return CopyStore(tmp_path / "copies.db")


def _ids_holders(snap):
    return {a["id"]: a["holder"] for a in snap["annotations"]}


# ---------- 记录规则 ----------

def test_create_records_revision_1_snapshot(store):
    snap = store.create(TEXT, [ann("a", 0, 3), ann("b", 2, 5)])
    cid = snap["id"]
    assert store.list_revisions(cid) == [
        {"revision": 1, "op": "create", "reviewer": None}]
    rev1 = store.get_revision(cid, 1)
    assert rev1["revision"] == 1 and rev1["status"] == "open"
    assert rev1["op"] == "create" and rev1["reviewer"] is None
    assert rev1["text"] == TEXT
    assert _ids_holders(rev1) == {"a": None, "b": None}
    assert rev1["groups"] == [{"annotation_ids": ["a", "b"], "holder": None}]


def test_each_mutation_records_full_snapshot(store):
    cid = store.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7),
                              ann("c", 12, 15)])["id"]
    store.claim(cid, "a", "alice")     # rev 2：整组 {a, b} 归 alice
    store.claim(cid, "c", "bob")       # rev 3
    store.modify(cid, "c", "bob", 13, 15, "丙改")  # rev 4
    store.release(cid, "b", "alice")   # rev 5：整组释放
    store.release(cid, "c", "bob")     # rev 6
    store.finalize(cid, 6)             # rev 7：结稿不携带 reviewer

    assert store.list_revisions(cid) == [
        {"revision": 1, "op": "create", "reviewer": None},
        {"revision": 2, "op": "claim", "reviewer": "alice"},
        {"revision": 3, "op": "claim", "reviewer": "bob"},
        {"revision": 4, "op": "modify", "reviewer": "bob"},
        {"revision": 5, "op": "release", "reviewer": "alice"},
        {"revision": 6, "op": "release", "reviewer": "bob"},
        {"revision": 7, "op": "finalize", "reviewer": None},
    ]

    rev2 = store.get_revision(cid, 2)
    assert _ids_holders(rev2) == {"a": "alice", "b": "alice", "c": None}
    assert rev2["status"] == "open"

    rev4 = store.get_revision(cid, 4)  # 修改后的版本：范围 / label / 持有权
    c4 = [a for a in rev4["annotations"] if a["id"] == "c"][0]
    assert (c4["start"], c4["end"], c4["label"], c4["holder"]) == \
        (13, 15, "丙改", "bob")
    assert _ids_holders(rev4) == {"a": "alice", "b": "alice", "c": "bob"}

    rev5 = store.get_revision(cid, 5)  # 释放 {a, b} 后：c 仍由 bob 持有
    assert _ids_holders(rev5) == {"a": None, "b": None, "c": "bob"}

    rev7 = store.get_revision(cid, 7)  # 结稿版本：仅 status 翻转，其余不变
    assert rev7["status"] == "finalized"
    assert rev7["op"] == "finalize" and rev7["reviewer"] is None
    assert _ids_holders(rev7) == {"a": None, "b": None, "c": None}
    c7 = [a for a in rev7["annotations"] if a["id"] == "c"][0]
    assert (c7["start"], c7["end"], c7["label"]) == (13, 15, "丙改")
    # 各版本快照互不影响：回看早期版本不受后续修改影响
    c2 = [a for a in rev2["annotations"] if a["id"] == "c"][0]
    assert (c2["start"], c2["end"], c2["label"]) == (12, 15, "L")


def test_snapshot_preserves_original_annotation_order(store):
    # 回看快照按该版保存的原序（创建顺序）返回批注，不重排为几何顺序
    cid = store.create(TEXT, [ann("z", 20, 25), ann("a", 0, 3),
                              ann("m", 10, 12)])["id"]
    rev1 = store.get_revision(cid, 1)
    assert [a["id"] for a in rev1["annotations"]] == ["z", "a", "m"]
    # 当前副本的读取仍按 (start, end, label, id) 几何排序
    assert [a["id"] for a in store.get(cid)["annotations"]] == ["a", "m", "z"]
    # 修改改变几何后，回看仍按该版保存的原序返回，字段来自所查版本
    store.claim(cid, "z", "alice")
    store.modify(cid, "z", "alice", 20, 24, "Z2")
    rev3 = store.get_revision(cid, 3)
    assert [a["id"] for a in rev3["annotations"]] == ["z", "a", "m"]
    z = [a for a in rev3["annotations"] if a["id"] == "z"][0]
    assert (z["start"], z["end"], z["label"], z["holder"]) == \
        (20, 24, "Z2", "alice")
    # 数据库中保存的原始排序（pos）即创建顺序
    conn = sqlite3.connect(store.path)
    try:
        rows = conn.execute(
            "SELECT ann_id FROM revision_annotations "
            "WHERE copy_id=? AND revision=1 ORDER BY pos", (cid,)).fetchall()
    finally:
        conn.close()
    assert [r[0] for r in rows] == ["z", "a", "m"]


def test_idempotent_claim_records_nothing(store):
    cid = store.create(TEXT, [ann("a", 0, 3)])["id"]
    store.claim(cid, "a", "alice")
    store.claim(cid, "a", "alice")  # 幂等：不增版本、不产生记录
    assert store.get(cid)["revision"] == 2
    assert store.list_revisions(cid) == [
        {"revision": 1, "op": "create", "reviewer": None},
        {"revision": 2, "op": "claim", "reviewer": "alice"}]


def test_failed_requests_record_nothing(store):
    cid = store.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7),
                              ann("c", 12, 15)])["id"]
    store.claim(cid, "a", "alice")   # rev 2
    store.claim(cid, "c", "bob")     # rev 3
    with pytest.raises(GroupHeldError):       # 他人持有的组
        store.claim(cid, "b", "bob")
    with pytest.raises(ModifyConflictError):  # 修改触及他人
        store.modify(cid, "a", "alice", 0, 13, "X")
    with pytest.raises(CopyStillHeldError):   # 仍有持有者
        store.finalize(cid, 3)
    with pytest.raises(RevisionMismatchError):  # 版本过期
        store.finalize(cid, 1)
    assert store.get(cid)["revision"] == 3
    assert [r["revision"] for r in store.list_revisions(cid)] == [1, 2, 3]


def test_snapshot_and_copy_change_commit_together(store, monkeypatch):
    # 快照写入失败 → 副本变更一并回滚，二者均不变
    cid = store.create(TEXT, [ann("a", 0, 3)])["id"]

    def boom(conn, state, op, reviewer):
        raise sqlite3.OperationalError("模拟快照写入失败")

    monkeypatch.setattr(store, "_record_revision", boom)
    with pytest.raises(sqlite3.OperationalError):
        store.claim(cid, "a", "alice")
    snap = store.get(cid)
    assert snap["revision"] == 1
    assert _ids_holders(snap) == {"a": None}
    monkeypatch.undo()
    assert store.list_revisions(cid) == [
        {"revision": 1, "op": "create", "reviewer": None}]
    # 恢复后正常变更仍从 revision=2 继续记录
    store.claim(cid, "a", "alice")
    assert [r["revision"] for r in store.list_revisions(cid)] == [1, 2]


# ---------- 并发：快照不混入后续版本 ----------

def test_concurrent_snapshots_never_mix_later_versions(store):
    # 10 个互不重叠的批注并行修改：每个版本的快照必须恰好是某次串行提交
    # 后的状态（前 r-1 次修改），不得混入更晚版本的内容
    anns = [ann(f"a{i}", 3 * i, 3 * i + 2, "L") for i in range(10)]
    cid = store.create(TEXT, anns)["id"]
    for i in range(10):
        store.claim(cid, f"a{i}", f"r{i}")
    base_rev = store.get(cid)["revision"]  # 1 + 10 次领取 = 11

    barrier = threading.Barrier(10)

    def edit(i):
        barrier.wait()
        store.modify(cid, f"a{i}", f"r{i}", 3 * i, 3 * i + 1, f"lab{i}")

    threads = [threading.Thread(target=edit, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert store.get(cid)["revision"] == base_rev + 10
    history = store.list_revisions(cid)
    assert [r["revision"] for r in history] == list(range(1, base_rev + 11))
    assert [r["op"] for r in history[-10:]] == ["modify"] * 10

    def modified_labels(rev):
        snap = store.get_revision(cid, rev)
        return {a["id"] for a in snap["annotations"]
                if a["label"].startswith("lab")}

    prev = set()
    for rev in range(base_rev + 1, base_rev + 11):
        current = modified_labels(rev)
        assert len(current) == rev - base_rev  # 恰好前 rev-base_rev 次修改
        assert prev <= current                 # 单调前缀：不丢、不混入未来
        prev = current
    assert len(prev) == 10


# ---------- 只读回看与 404 ----------

def test_reading_history_does_not_change_current_state(store):
    cid = store.create(TEXT, [ann("a", 0, 3)])["id"]
    store.claim(cid, "a", "alice")
    before = store.get(cid)
    store.list_revisions(cid)
    store.get_revision(cid, 1)
    store.get_revision(cid, 2)
    assert store.get(cid) == before


def test_unknown_copy_and_revision_404(store):
    with pytest.raises(CopyNotFound):
        store.list_revisions("deadbeef")
    with pytest.raises(CopyNotFound):
        store.get_revision("deadbeef", 1)
    cid = store.create(TEXT, [ann("a", 0, 3)])["id"]
    store.claim(cid, "a", "alice")  # 已记录 1、2
    for bad in (0, 3, 99, -1):
        with pytest.raises(RevisionNotFound):
            store.get_revision(cid, bad)


def test_history_survives_reopen(tmp_path):
    db = tmp_path / "copies.db"
    s1 = CopyStore(db)
    cid = s1.create(TEXT, [ann("a", 0, 3)])["id"]
    s1.claim(cid, "a", "alice")
    s1.release(cid, "a", "alice")
    s1.finalize(cid, 3)

    s2 = CopyStore(db)  # 模拟容器重启：历史随原库保留
    assert [r["op"] for r in s2.list_revisions(cid)] == \
        ["create", "claim", "release", "finalize"]
    rev2 = s2.get_revision(cid, 2)
    assert _ids_holders(rev2) == {"a": "alice"}
    assert s2.get_revision(cid, 4)["status"] == "finalized"


# ---------- 旧库 baseline 补记 ----------

def _make_old_db(db):
    """手工构造无历史表（也无 status 列）的旧库，含一个 revision=5 的副本。"""
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE copies (
            id TEXT PRIMARY KEY, text TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE annotations (
            copy_id TEXT NOT NULL, ann_id TEXT NOT NULL, pos INTEGER NOT NULL,
            start INTEGER NOT NULL, end INTEGER NOT NULL, label TEXT NOT NULL,
            holder TEXT, PRIMARY KEY (copy_id, ann_id));
    """)
    conn.execute("INSERT INTO copies(id, text, revision) VALUES('c1', ?, 5)",
                 (TEXT,))
    conn.execute("INSERT INTO annotations VALUES('c1', 'a', 0, 0, 3, 'L', NULL)")
    conn.execute(
        "INSERT INTO annotations VALUES('c1', 'b', 1, 4, 7, 'L', 'alice')")
    conn.commit()
    conn.close()


def test_old_db_backfills_baseline_only(tmp_path):
    db = tmp_path / "copies.db"
    _make_old_db(db)

    store = CopyStore(db)
    # 仅补当前 revision 的一条 baseline：不编造 1..4 的早期历史
    assert store.list_revisions("c1") == [
        {"revision": 5, "op": "baseline", "reviewer": None}]
    base = store.get_revision("c1", 5)
    assert base["status"] == "open"  # 旧副本默认 open
    assert base["text"] == TEXT
    assert _ids_holders(base) == {"a": None, "b": "alice"}  # 原持有权不变
    with pytest.raises(RevisionNotFound):
        store.get_revision("c1", 1)  # 不编造早期历史
    with pytest.raises(RevisionNotFound):
        store.get_revision("c1", 4)
    # 不改原数据：revision 仍为 5
    assert store.get("c1")["revision"] == 5

    # 重复启动不重复补记
    again = CopyStore(db)
    assert again.list_revisions("c1") == [
        {"revision": 5, "op": "baseline", "reviewer": None}]

    # 补记后的新变更照常记录（从 revision=6 起）
    again.release("c1", "b", "alice")
    assert [r["op"] for r in again.list_revisions("c1")] == \
        ["baseline", "release"]
    rev6 = again.get_revision("c1", 6)
    assert rev6["op"] == "release" and rev6["reviewer"] == "alice"
    assert _ids_holders(rev6) == {"a": None, "b": None}


def test_baseline_backfill_skips_copies_with_history(tmp_path):
    # 新功能创建的副本自带 create 记录，重启不会被补 baseline
    db = tmp_path / "copies.db"
    s1 = CopyStore(db)
    cid = s1.create(TEXT, [ann("a", 0, 3)])["id"]
    s2 = CopyStore(db)
    assert [r["op"] for r in s2.list_revisions(cid)] == ["create"]


# ---------- HTTP 端到端 ----------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("COPY_DB_PATH", str(tmp_path / "copies.db"))
    import app.main as main
    importlib.reload(main)  # 用新环境变量重新初始化全局仓库
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c
    main._store = None


def _create(client, text=TEXT, anns=None):
    r = client.post("/api/copies", json={"text": text,
                                         "annotations": anns or []})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_http_history_full_flow(client):
    cid = _create(client, anns=[ann("a", 0, 4), ann("b", 3, 7),
                                ann("c", 12, 15)])
    client.post(f"/api/copies/{cid}/annotations/a/claim",
                json={"reviewer": "alice"})
    client.post(f"/api/copies/{cid}/annotations/c/claim",
                json={"reviewer": "bob"})
    client.patch(f"/api/copies/{cid}/annotations/c",
                 json={"reviewer": "bob", "start": 13, "end": 15,
                       "label": "丙改"})
    client.post(f"/api/copies/{cid}/annotations/b/release",
                json={"reviewer": "alice"})
    client.post(f"/api/copies/{cid}/annotations/c/release",
                json={"reviewer": "bob"})
    r = client.post(f"/api/copies/{cid}/finalize",
                    json={"expected_revision": 6})
    assert r.status_code == 200

    # 版本列表：按 revision 升序，含操作类型与 reviewer
    r = client.get(f"/api/copies/{cid}/revisions")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == cid
    assert body["revisions"] == [
        {"revision": 1, "op": "create", "reviewer": None},
        {"revision": 2, "op": "claim", "reviewer": "alice"},
        {"revision": 3, "op": "claim", "reviewer": "bob"},
        {"revision": 4, "op": "modify", "reviewer": "bob"},
        {"revision": 5, "op": "release", "reviewer": "alice"},
        {"revision": 6, "op": "release", "reviewer": "bob"},
        {"revision": 7, "op": "finalize", "reviewer": None},
    ]

    # 读取指定版本完整快照：revision=4 时 c 已被 bob 改过
    r = client.get(f"/api/copies/{cid}/revisions/4")
    assert r.status_code == 200
    snap = r.json()
    assert snap["id"] == cid and snap["revision"] == 4
    assert snap["op"] == "modify" and snap["reviewer"] == "bob"
    assert snap["status"] == "open" and snap["text"] == TEXT
    by_id = {a["id"]: a for a in snap["annotations"]}
    assert (by_id["c"]["start"], by_id["c"]["end"], by_id["c"]["label"],
            by_id["c"]["holder"]) == (13, 15, "丙改", "bob")
    assert by_id["a"]["holder"] == "alice"
    assert snap["groups"]  # 分组按该版本几何重算

    # 回看 revision=1：创建时无人持有
    rev1 = client.get(f"/api/copies/{cid}/revisions/1").json()
    assert all(a["holder"] is None for a in rev1["annotations"])
    assert rev1["op"] == "create" and rev1["reviewer"] is None

    # 回看结稿版本
    rev7 = client.get(f"/api/copies/{cid}/revisions/7").json()
    assert rev7["status"] == "finalized" and rev7["reviewer"] is None

    # 只读：当前状态不受回看影响
    current = client.get(f"/api/copies/{cid}").json()
    assert current["revision"] == 7 and current["status"] == "finalized"


def test_http_history_unknown_copy_and_revision_404(client):
    r = client.get("/api/copies/deadbeef/revisions")
    assert r.status_code == 404
    r = client.get("/api/copies/deadbeef/revisions/1")
    assert r.status_code == 404

    cid = _create(client, anns=[ann("a", 0, 3)])
    client.post(f"/api/copies/{cid}/annotations/a/claim",
                json={"reviewer": "alice"})
    for bad in (0, 3, 99):
        r = client.get(f"/api/copies/{cid}/revisions/{bad}")
        assert r.status_code == 404, bad


def test_http_failed_and_idempotent_requests_leave_no_history(client):
    cid = _create(client, anns=[ann("a", 0, 4), ann("b", 3, 7)])
    client.post(f"/api/copies/{cid}/annotations/a/claim",
                json={"reviewer": "alice"})
    # 幂等重复领取：200 但不产生新版本
    r = client.post(f"/api/copies/{cid}/annotations/a/claim",
                    json={"reviewer": "alice"})
    assert r.status_code == 200 and r.json()["revision"] == 2
    # 各类失败请求：409 / 422 / 404 均不产生记录
    assert client.post(f"/api/copies/{cid}/annotations/b/claim",
                       json={"reviewer": "bob"}).status_code == 409
    assert client.post(f"/api/copies/{cid}/annotations/a/claim",
                       json={"reviewer": ""}).status_code == 422
    assert client.post(f"/api/copies/{cid}/annotations/nope/claim",
                       json={"reviewer": "bob"}).status_code == 404
    assert client.post(f"/api/copies/{cid}/finalize",
                       json={"expected_revision": 2}).status_code == 409
    body = client.get(f"/api/copies/{cid}/revisions").json()
    assert body["revisions"] == [
        {"revision": 1, "op": "create", "reviewer": None},
        {"revision": 2, "op": "claim", "reviewer": "alice"}]
    assert client.get(f"/api/copies/{cid}").json()["revision"] == 2


# ---------- HTTP 回归：回看按该版保存的原序返回批注 ----------

# 创建顺序（z → a → m）与几何 (start,end,label,id) 顺序（a → m → z）不同
SHUFFLED_ANNS = [ann("z", 20, 25, "丁"), ann("a", 0, 3, "甲"),
                 ann("m", 10, 12, "丙")]


def test_http_revision_returns_saved_creation_order(client):
    r = client.post("/api/copies", json={"text": TEXT,
                                         "annotations": SHUFFLED_ANNS})
    assert r.status_code == 201
    cid = r.json()["id"]
    # 创建响应（当前副本）按 (start,end,label,id) 几何排序
    assert [a["id"] for a in r.json()["annotations"]] == ["a", "m", "z"]

    # 回看 revision=1：按该版保存的原序（创建顺序）返回，字段来自该版
    rev1 = client.get(f"/api/copies/{cid}/revisions/1").json()
    assert rev1["op"] == "create" and rev1["reviewer"] is None
    assert [a["id"] for a in rev1["annotations"]] == ["z", "a", "m"]
    assert [(a["start"], a["end"], a["label"], a["holder"])
            for a in rev1["annotations"]] == \
        [(20, 25, "丁", None), (0, 3, "甲", None), (10, 12, "丙", None)]

    # 当前副本的读取仍按几何排序；回看只读，不改变当前状态
    current = client.get(f"/api/copies/{cid}").json()
    assert [a["id"] for a in current["annotations"]] == ["a", "m", "z"]
    assert current["revision"] == 1


def test_http_revision_keeps_saved_order_after_modify(client):
    cid = _create(client, anns=SHUFFLED_ANNS)
    client.post(f"/api/copies/{cid}/annotations/z/claim",
                json={"reviewer": "alice"})
    # 修改把 z 移到 a 与 m 之间的几何位置（不与任何批注重叠）
    r = client.patch(f"/api/copies/{cid}/annotations/z",
                     json={"reviewer": "alice", "start": 4, "end": 6,
                           "label": "丁改"})
    assert r.status_code == 200
    # 变更当前副本的响应按新几何排序
    assert [a["id"] for a in r.json()["annotations"]] == ["a", "z", "m"]

    # 回看修改版本（revision=3）：仍按该版保存的原序，字段来自该版
    rev3 = client.get(f"/api/copies/{cid}/revisions/3").json()
    assert rev3["op"] == "modify" and rev3["reviewer"] == "alice"
    assert [a["id"] for a in rev3["annotations"]] == ["z", "a", "m"]
    z = rev3["annotations"][0]
    assert (z["start"], z["end"], z["label"], z["holder"]) == \
        (4, 6, "丁改", "alice")

    # 回看创建版本：原范围、无人持有，顺序同为创建顺序
    rev1 = client.get(f"/api/copies/{cid}/revisions/1").json()
    assert [a["id"] for a in rev1["annotations"]] == ["z", "a", "m"]
    assert (rev1["annotations"][0]["start"], rev1["annotations"][0]["end"],
            rev1["annotations"][0]["label"]) == (20, 25, "丁")

    # 只读：当前副本不受回看影响，仍按新几何排序
    current = client.get(f"/api/copies/{cid}").json()
    assert current["revision"] == 3
    assert [a["id"] for a in current["annotations"]] == ["a", "z", "m"]


def _make_shuffled_old_db(db):
    """手工构造无历史表的旧库：批注 pos 顺序（z → a → m）与几何排序不同。"""
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE copies (
            id TEXT PRIMARY KEY, text TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE annotations (
            copy_id TEXT NOT NULL, ann_id TEXT NOT NULL, pos INTEGER NOT NULL,
            start INTEGER NOT NULL, end INTEGER NOT NULL, label TEXT NOT NULL,
            holder TEXT, PRIMARY KEY (copy_id, ann_id));
    """)
    conn.execute("INSERT INTO copies(id, text, revision) VALUES('c1', ?, 5)",
                 (TEXT,))
    conn.execute(
        "INSERT INTO annotations VALUES('c1', 'z', 0, 20, 25, '丁', NULL)")
    conn.execute(
        "INSERT INTO annotations VALUES('c1', 'a', 1, 0, 3, '甲', 'alice')")
    conn.execute(
        "INSERT INTO annotations VALUES('c1', 'm', 2, 10, 12, '丙', NULL)")
    conn.commit()
    conn.close()


def test_http_revision_baseline_keeps_old_db_order(tmp_path, monkeypatch):
    # 旧库（无历史表）：启动时补记的 baseline 回看同样按保存的原序返回
    db = tmp_path / "copies.db"
    _make_shuffled_old_db(db)
    monkeypatch.setenv("COPY_DB_PATH", str(db))
    import app.main as main
    importlib.reload(main)  # 用新环境变量重新初始化全局仓库
    from fastapi.testclient import TestClient
    with TestClient(main.app) as client:
        body = client.get("/api/copies/c1/revisions").json()
        assert body["revisions"] == [
            {"revision": 5, "op": "baseline", "reviewer": None}]
        base = client.get("/api/copies/c1/revisions/5").json()
        assert base["op"] == "baseline" and base["reviewer"] is None
        assert [a["id"] for a in base["annotations"]] == ["z", "a", "m"]
        assert base["annotations"][1]["holder"] == "alice"  # 字段来自该版
        # 当前副本读取仍按几何排序；不回写历史数据（重复启动不重复补记）
        current = client.get("/api/copies/c1").json()
        assert [a["id"] for a in current["annotations"]] == ["a", "m", "z"]
    main._store = None


def test_http_revision_order_survives_restart(client):
    import app.main as main
    cid = _create(client, anns=SHUFFLED_ANNS)
    client.post(f"/api/copies/{cid}/annotations/z/claim",
                json={"reviewer": "alice"})

    # 模拟容器重启：丢弃内存中的仓库，下次请求在同一数据库文件上重建
    main._store = None
    rev1 = client.get(f"/api/copies/{cid}/revisions/1").json()
    assert [a["id"] for a in rev1["annotations"]] == ["z", "a", "m"]
    rev2 = client.get(f"/api/copies/{cid}/revisions/2").json()
    assert rev2["op"] == "claim" and rev2["reviewer"] == "alice"
    assert [a["id"] for a in rev2["annotations"]] == ["z", "a", "m"]
    assert rev2["annotations"][0]["holder"] == "alice"
    # 重启后当前副本读取仍按几何排序
    current = client.get(f"/api/copies/{cid}").json()
    assert [a["id"] for a in current["annotations"]] == ["a", "m", "z"]
