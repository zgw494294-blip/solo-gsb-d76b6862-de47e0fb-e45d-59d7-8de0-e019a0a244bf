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
    # 批注原排序（创建顺序）须按版本保留，而不是按 start 排序：
    # 回看每个版本都按该版保存的原序返回，即使原序不同于几何顺序
    cid = store.create(TEXT, [ann("z", 20, 25), ann("a", 0, 3),
                              ann("m", 10, 12)])["id"]
    rev1 = store.get_revision(cid, 1)
    assert [a["id"] for a in rev1["annotations"]] == ["z", "a", "m"]
    # groups 仍按该版几何稳定计算（与批注列表的排列无关）
    assert [g["annotation_ids"] for g in rev1["groups"]] == \
        [["a"], ["m"], ["z"]]

    store.claim(cid, "z", "alice")
    rev2 = store.get_revision(cid, 2)
    assert [a["id"] for a in rev2["annotations"]] == ["z", "a", "m"]
    assert _ids_holders(rev2) == {"z": "alice", "a": None, "m": None}

    store.modify(cid, "z", "alice", 20, 24, "Z2")
    rev3 = store.get_revision(cid, 3)
    # 修改版本同样按原序（创建顺序）返回，不重排为几何顺序
    assert [a["id"] for a in rev3["annotations"]] == ["z", "a", "m"]
    z = rev3["annotations"][0]
    assert (z["start"], z["end"], z["label"], z["holder"]) == \
        (20, 24, "Z2", "alice")
    # 回看不改当前副本：当前副本读取仍按 (start,end,label,id) 几何排序
    current = store.get(cid)
    assert [a["id"] for a in current["annotations"]] == ["a", "m", "z"]
    # 早期版本不受后续修改影响，原序与原几何都保留
    z1 = rev1["annotations"][0]
    assert (z1["start"], z1["end"], z1["label"], z1["holder"]) == \
        (20, 25, "L", None)
    # 数据库中保存的原始排序（pos）即创建顺序；回看前后均不被重写
    def _stored_pos(rev):
        conn = sqlite3.connect(store.path)
        try:
            return [r[0] for r in conn.execute(
                "SELECT ann_id FROM revision_annotations "
                "WHERE copy_id=? AND revision=? ORDER BY pos",
                (cid, rev)).fetchall()]
        finally:
            conn.close()

    assert _stored_pos(1) == ["z", "a", "m"]
    assert _stored_pos(3) == ["z", "a", "m"]
    store.get_revision(cid, 1)
    store.get_revision(cid, 2)
    store.get_revision(cid, 3)
    assert _stored_pos(1) == ["z", "a", "m"]
    assert _stored_pos(3) == ["z", "a", "m"]


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
    """手工构造无历史表（也无 status 列）的旧库，含一个 revision=5 的副本。

    annotations 的 pos 原序（b 在前、a 在后）与几何顺序（a 在前）不同，
    用于验证 baseline 回看按原序返回、不重写历史数据。
    """
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
    # pos=0 的 b 在文本后部；pos=1 的 a 在开头——原序不同于几何顺序
    conn.execute(
        "INSERT INTO annotations VALUES('c1', 'b', 0, 4, 7, 'L', 'alice')")
    conn.execute(
        "INSERT INTO annotations VALUES('c1', 'a', 1, 0, 3, 'L', NULL)")
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
    # baseline 回看按旧库保存的原序（pos）返回，不重排为几何顺序
    assert [a["id"] for a in base["annotations"]] == ["b", "a"]
    assert [g["annotation_ids"] for g in base["groups"]] == [["a"], ["b"]]
    # 当前副本读取仍按几何顺序
    assert [a["id"] for a in store.get("c1")["annotations"]] == ["a", "b"]
    with pytest.raises(RevisionNotFound):
        store.get_revision("c1", 1)  # 不编造早期历史
    with pytest.raises(RevisionNotFound):
        store.get_revision("c1", 4)
    # 不改原数据：revision 仍为 5
    assert store.get("c1")["revision"] == 5

    # 重复启动不重复补记；已写入的 baseline 原序不被重写
    again = CopyStore(db)
    assert again.list_revisions("c1") == [
        {"revision": 5, "op": "baseline", "reviewer": None}]
    assert [a["id"] for a in again.get_revision("c1", 5)["annotations"]] == \
        ["b", "a"]

    # 补记后的新变更照常记录（从 revision=6 起）
    again.release("c1", "b", "alice")
    assert [r["op"] for r in again.list_revisions("c1")] == \
        ["baseline", "release"]
    rev6 = again.get_revision("c1", 6)
    assert rev6["op"] == "release" and rev6["reviewer"] == "alice"
    assert _ids_holders(rev6) == {"a": None, "b": None}
    # 新版本同样继承原序；回看不改变当前副本的几何排序输出
    assert [a["id"] for a in rev6["annotations"]] == ["b", "a"]
    assert [a["id"] for a in again.get("c1")["annotations"]] == ["a", "b"]


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


# ---------- 回归：原序不同于几何顺序时，回看按原序、当前按几何序 ----------

def _ann_order(snap):
    return [a["id"] for a in snap["annotations"]]


def test_http_replay_returns_saved_order_not_geometric(client):
    # 创建顺序 z, a, m 与几何顺序 a, m, z 不同
    cid = _create(client, anns=[ann("z", 20, 25), ann("a", 0, 3),
                                ann("m", 10, 12)])

    # 创建响应（当前副本）继续按 (start,end,label,id) 几何排序
    created = client.post("/api/copies", json={
        "text": TEXT,
        "annotations": [ann("z2", 21, 24), ann("a2", 1, 2), ann("m2", 11, 12)],
    })
    assert created.status_code == 201
    assert _ann_order(created.json()) == ["a2", "m2", "z2"]
    other = created.json()["id"]

    # 读取当前副本也是几何序
    assert _ann_order(client.get(f"/api/copies/{cid}").json()) == \
        ["a", "m", "z"]

    # 回看创建版本：按该版保存的原序（创建顺序）返回
    rev1 = client.get(f"/api/copies/{cid}/revisions/1")
    assert rev1.status_code == 200
    body = rev1.json()
    assert _ann_order(body) == ["z", "a", "m"]
    assert all(a["holder"] is None for a in body["annotations"])
    # groups 仍按该版几何稳定计算
    assert [g["annotation_ids"] for g in body["groups"]] == \
        [["a"], ["m"], ["z"]]

    # 另一个乱序副本的回看同样按创建原序（证明创建路径没有改写历史）
    assert _ann_order(
        client.get(f"/api/copies/{other}/revisions/1").json()) == \
        ["z2", "a2", "m2"]


def test_http_replay_after_modify_keeps_saved_order_per_revision(client):
    # 创建顺序 c, a, b；几何顺序 a, b, c
    cid = _create(client, anns=[ann("c", 12, 15), ann("a", 0, 4),
                                ann("b", 3, 7)])

    assert client.post(f"/api/copies/{cid}/annotations/a/claim",
                       json={"reviewer": "alice"}).status_code == 200
    assert client.post(f"/api/copies/{cid}/annotations/c/claim",
                       json={"reviewer": "bob"}).status_code == 200
    r = client.patch(f"/api/copies/{cid}/annotations/c",
                     json={"reviewer": "bob", "start": 13, "end": 15,
                           "label": "丙改"})
    assert r.status_code == 200

    # 变更当前副本的响应继续按几何排序
    assert _ann_order(r.json()) == ["a", "b", "c"]
    assert _ann_order(client.get(f"/api/copies/{cid}").json()) == \
        ["a", "b", "c"]

    # 每个历史版本都按保存的原序返回；文本 / 范围 / label / holder / status /
    # revision / op / reviewer 均来自所查版本
    rev1 = client.get(f"/api/copies/{cid}/revisions/1").json()
    assert _ann_order(rev1) == ["c", "a", "b"]
    assert rev1["revision"] == 1 and rev1["op"] == "create"
    assert rev1["status"] == "open" and rev1["reviewer"] is None
    by_id = {a["id"]: a for a in rev1["annotations"]}
    assert (by_id["c"]["start"], by_id["c"]["end"], by_id["c"]["label"],
            by_id["c"]["holder"]) == (12, 15, "L", None)

    rev2 = client.get(f"/api/copies/{cid}/revisions/2").json()
    assert _ann_order(rev2) == ["c", "a", "b"]
    assert rev2["op"] == "claim" and rev2["reviewer"] == "alice"
    by_id = {a["id"]: a for a in rev2["annotations"]}
    assert by_id["a"]["holder"] == by_id["b"]["holder"] == "alice"
    assert by_id["c"]["holder"] is None

    rev3 = client.get(f"/api/copies/{cid}/revisions/3").json()
    assert _ann_order(rev3) == ["c", "a", "b"]
    by_id = {a["id"]: a for a in rev3["annotations"]}
    assert by_id["c"]["holder"] == "bob"
    assert (by_id["c"]["start"], by_id["c"]["end"],
            by_id["c"]["label"]) == (12, 15, "L")

    rev4 = client.get(f"/api/copies/{cid}/revisions/4").json()
    assert rev4["op"] == "modify" and rev4["reviewer"] == "bob"
    assert _ann_order(rev4) == ["c", "a", "b"]
    by_id = {a["id"]: a for a in rev4["annotations"]}
    assert (by_id["c"]["start"], by_id["c"]["end"], by_id["c"]["label"],
            by_id["c"]["holder"]) == (13, 15, "丙改", "bob")
    assert [g["annotation_ids"] for g in rev4["groups"]] == \
        [["a", "b"], ["c"]]

    # 回看不改当前副本：revision、几何序输出与修改后内容保持不变
    current = client.get(f"/api/copies/{cid}").json()
    assert current["revision"] == 4
    assert _ann_order(current) == ["a", "b", "c"]


def test_http_replay_old_db_baseline_keeps_pos_order(tmp_path, monkeypatch):
    # 旧库（无历史表 / 无 status 列）：annotations 的 pos 原序不同于几何序
    db = tmp_path / "copies.db"
    _make_old_db(db)
    monkeypatch.setenv("COPY_DB_PATH", str(db))
    import app.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        # baseline 回看按旧库 pos 原序返回（覆盖已存 baseline 不重写历史）
        r = c.get("/api/copies/c1/revisions/5")
        assert r.status_code == 200
        body = r.json()
        assert body["op"] == "baseline" and body["reviewer"] is None
        assert _ann_order(body) == ["b", "a"]
        by_id = {a["id"]: a for a in body["annotations"]}
        assert by_id["b"]["holder"] == "alice"
        assert by_id["a"]["holder"] is None
        # groups 按该版几何：a 在 b 之前且二者不重叠
        assert [g["annotation_ids"] for g in body["groups"]] == \
            [["a"], ["b"]]
        # 当前副本读取仍是几何序，revision 不被回看改变
        current = c.get("/api/copies/c1").json()
        assert _ann_order(current) == ["a", "b"]
        assert current["revision"] == 5

        # 重启后再看：baseline 不重复补记、原序不被重写
        main._store = None
        with TestClient(main.app) as again:
            r2 = again.get("/api/copies/c1/revisions/5")
            assert r2.status_code == 200
            assert _ann_order(r2.json()) == ["b", "a"]
            assert again.get("/api/copies/c1/revisions").json()[
                "revisions"] == [
                {"revision": 5, "op": "baseline", "reviewer": None}]
    main._store = None


def test_http_replay_keeps_saved_order_after_restart(tmp_path, monkeypatch):
    db = tmp_path / "copies.db"
    monkeypatch.setenv("COPY_DB_PATH", str(db))
    import app.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        r = c.post("/api/copies", json={
            "text": TEXT,
            "annotations": [ann("z", 20, 25), ann("a", 0, 3),
                            ann("m", 10, 12)],
        })
        cid = r.json()["id"]
        c.post(f"/api/copies/{cid}/annotations/z/claim",
               json={"reviewer": "alice"})
        c.patch(f"/api/copies/{cid}/annotations/z",
                json={"reviewer": "alice", "start": 20, "end": 24,
                      "label": "Z2"})
        assert c.get(f"/api/copies/{cid}").json()["revision"] == 3

    # 模拟容器重启：重新初始化仓库（同一数据卷 / 数据库文件）
    main._store = None
    with TestClient(main.app) as c:
        for rev, expected in ((1, ["z", "a", "m"]),
                              (2, ["z", "a", "m"]),
                              (3, ["z", "a", "m"])):
            body = c.get(f"/api/copies/{cid}/revisions/{rev}").json()
            assert _ann_order(body) == expected
        rev3 = c.get(f"/api/copies/{cid}/revisions/3").json()
        z = rev3["annotations"][0]
        assert (z["start"], z["end"], z["label"], z["holder"]) == \
            (20, 24, "Z2", "alice")
        # 当前副本仍按几何序
        assert _ann_order(c.get(f"/api/copies/{cid}").json()) == \
            ["a", "m", "z"]
    main._store = None
