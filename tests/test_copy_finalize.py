"""工作副本结稿（finalize）：HTTP 接口、SQLite 持久化 / 迁移与并发测试。"""

import importlib
import sqlite3
import threading

import pytest

from app.copies import (
    CopyFinalizedError,
    CopyState,
    CopyStillHeldError,
    CopyStore,
    CopyValidationError,
    RevisionMismatchError,
)

TEXT = "x" * 30


def ann(id, start, end, label="L"):
    return {"id": id, "start": start, "end": end, "label": label}


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


def _finalize(client, cid, expected_revision):
    return client.post(f"/api/copies/{cid}/finalize",
                       json={"expected_revision": expected_revision})


def test_create_and_get_carry_status_open(client):
    cid = _create(client, anns=[ann("a", 0, 3)])
    created = client.get(f"/api/copies/{cid}").json()
    assert created["status"] == "open"
    r = client.post("/api/copies", json={"text": TEXT, "annotations": []})
    assert r.json()["status"] == "open"


def test_finalize_success_only_flips_status_and_bumps_revision(client):
    cid = _create(client, anns=[ann("a", 0, 4), ann("b", 3, 7),
                                ann("c", 12, 15)])
    client.post(f"/api/copies/{cid}/annotations/a/claim",
                json={"reviewer": "alice"})
    client.post(f"/api/copies/{cid}/annotations/a/release",
                json={"reviewer": "alice"})
    before = client.get(f"/api/copies/{cid}").json()
    assert before["revision"] == 3 and before["status"] == "open"

    r = _finalize(client, cid, 3)
    assert r.status_code == 200
    snap = r.json()
    assert snap["status"] == "finalized"
    assert snap["revision"] == 4
    # 文本、批注、持有权与排序均不变
    assert snap["text"] == before["text"]
    assert snap["annotations"] == before["annotations"]
    assert snap["groups"] == before["groups"]

    again = client.get(f"/api/copies/{cid}").json()
    assert again == snap


def test_finalize_stale_revision_409_and_unchanged(client):
    cid = _create(client, anns=[ann("a", 0, 3)])
    client.post(f"/api/copies/{cid}/annotations/a/claim",
                json={"reviewer": "alice"})
    client.post(f"/api/copies/{cid}/annotations/a/release",
                json={"reviewer": "alice"})  # revision 现为 3
    r = _finalize(client, cid, 2)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["expected_revision"] == 2 and detail["revision"] == 3
    snap = client.get(f"/api/copies/{cid}").json()
    assert snap["status"] == "open" and snap["revision"] == 3


def test_finalize_with_holders_409_and_unchanged(client):
    cid = _create(client, anns=[ann("a", 0, 4), ann("b", 3, 7)])
    client.post(f"/api/copies/{cid}/annotations/a/claim",
                json={"reviewer": "alice"})
    r = _finalize(client, cid, 2)
    assert r.status_code == 409
    assert r.json()["detail"]["holders"] == {"a": "alice", "b": "alice"}
    snap = client.get(f"/api/copies/{cid}").json()
    assert snap["status"] == "open" and snap["revision"] == 2
    assert {a["id"]: a["holder"] for a in snap["annotations"]} == \
        {"a": "alice", "b": "alice"}


def test_finalize_release_then_finalize_succeeds(client):
    cid = _create(client, anns=[ann("a", 0, 3)])
    client.post(f"/api/copies/{cid}/annotations/a/claim",
                json={"reviewer": "alice"})
    assert _finalize(client, cid, 2).status_code == 409  # 仍有持有者
    client.post(f"/api/copies/{cid}/annotations/a/release",
                json={"reviewer": "alice"})
    r = _finalize(client, cid, 3)
    assert r.status_code == 200
    assert r.json()["status"] == "finalized"


def test_finalize_already_finalized_409_no_reopen(client):
    cid = _create(client, anns=[ann("a", 0, 3)])
    assert _finalize(client, cid, 1).status_code == 200
    r = _finalize(client, cid, 2)  # 版本正确也已结稿
    assert r.status_code == 409
    snap = client.get(f"/api/copies/{cid}").json()
    assert snap["status"] == "finalized" and snap["revision"] == 2


def test_finalize_invalid_expected_revision_422(client):
    cid = _create(client, anns=[ann("a", 0, 3)])
    for body in ({"expected_revision": True}, {"expected_revision": False},
                 {"expected_revision": 0}, {"expected_revision": -2},
                 {"expected_revision": 1.5}, {"expected_revision": None},
                 {}):
        r = client.post(f"/api/copies/{cid}/finalize", json=body)
        assert r.status_code == 422, body
    snap = client.get(f"/api/copies/{cid}").json()
    assert snap["status"] == "open" and snap["revision"] == 1


def test_finalize_unknown_copy_404(client):
    r = _finalize(client, "deadbeef", 1)
    assert r.status_code == 404


def test_claim_release_modify_after_finalize_409(client):
    cid = _create(client, anns=[ann("a", 0, 4), ann("b", 3, 7),
                                ann("c", 12, 15)])
    client.post(f"/api/copies/{cid}/annotations/a/claim",
                json={"reviewer": "alice"})
    # 修改把 a 从组里拆开：碎片 b 保留原持有者 alice
    client.patch(f"/api/copies/{cid}/annotations/a",
                 json={"reviewer": "alice", "start": 0, "end": 2, "label": "X"})
    client.post(f"/api/copies/{cid}/annotations/a/release",
                json={"reviewer": "alice"})
    client.post(f"/api/copies/{cid}/annotations/b/release",
                json={"reviewer": "alice"})
    assert _finalize(client, cid, 5).status_code == 200

    r = client.post(f"/api/copies/{cid}/annotations/c/claim",
                    json={"reviewer": "bob"})
    assert r.status_code == 409
    r = client.post(f"/api/copies/{cid}/annotations/a/release",
                    json={"reviewer": "alice"})
    assert r.status_code == 409
    r = client.patch(f"/api/copies/{cid}/annotations/a",
                     json={"reviewer": "alice", "start": 0, "end": 3,
                           "label": "Y"})
    assert r.status_code == 409
    # 全部失败后副本保持结稿时的提交状态
    snap = client.get(f"/api/copies/{cid}").json()
    assert snap["status"] == "finalized" and snap["revision"] == 6
    by_id = {a["id"]: a for a in snap["annotations"]}
    assert (by_id["a"]["start"], by_id["a"]["end"], by_id["a"]["label"]) == \
        (0, 2, "X")
    assert all(a["holder"] is None for a in snap["annotations"])


def test_original_404_422_semantics_preserved_after_finalize(client):
    cid = _create(client, anns=[ann("a", 0, 3)])
    assert _finalize(client, cid, 1).status_code == 200
    # 未知批注仍 404；非法输入仍 422
    r = client.post(f"/api/copies/{cid}/annotations/nope/claim",
                    json={"reviewer": "alice"})
    assert r.status_code == 404
    r = client.patch(f"/api/copies/{cid}/annotations/nope",
                     json={"reviewer": "alice", "start": 0, "end": 1,
                           "label": "X"})
    assert r.status_code == 404
    r = client.post(f"/api/copies/{cid}/annotations/a/claim",
                    json={"reviewer": ""})
    assert r.status_code == 422
    r = client.patch(f"/api/copies/{cid}/annotations/a",
                     json={"reviewer": "alice", "start": 0, "end": 99,
                           "label": "X"})
    assert r.status_code == 422
    r = client.post("/api/copies/nope/annotations/a/claim",
                    json={"reviewer": "alice"})
    assert r.status_code == 404


# ---------- 纯内存核心 ----------

def test_state_finalize_success_and_guards():
    st = CopyState("c1", TEXT, [ann("a", 0, 3)])
    st.finalize(1)
    assert st.status == "finalized" and st.revision == 2
    snap = st.snapshot()
    assert snap["status"] == "finalized"
    with pytest.raises(CopyFinalizedError):
        st.claim("a", "alice")
    with pytest.raises(CopyFinalizedError):
        st.release("a", "alice")
    with pytest.raises(CopyFinalizedError):
        st.modify("a", "alice", 0, 2, "X")
    with pytest.raises(CopyFinalizedError):
        st.finalize(2)


def test_state_finalize_rejects_bad_expected_revision():
    st = CopyState("c1", TEXT, [ann("a", 0, 3)])
    for bad in (True, False, 0, -1, "1", 1.0, None):
        with pytest.raises(CopyValidationError):
            st.finalize(bad)
    assert st.status == "open" and st.revision == 1


def test_state_finalize_stale_and_held():
    st = CopyState("c1", TEXT, [ann("a", 0, 3)])
    st.claim("a", "alice")
    with pytest.raises(RevisionMismatchError):
        st.finalize(1)
    with pytest.raises(CopyStillHeldError):
        st.finalize(2)
    st.release("a", "alice")
    st.finalize(3)
    assert st.status == "finalized" and st.revision == 4


# ---------- SQLite 持久化 / 旧库迁移 ----------

@pytest.fixture()
def store(tmp_path):
    return CopyStore(tmp_path / "copies.db")


def test_finalize_persisted_across_reopen(tmp_path):
    db = tmp_path / "copies.db"
    s1 = CopyStore(db)
    cid = s1.create(TEXT, [ann("a", 0, 3)])["id"]
    s1.finalize(cid, 1)

    s2 = CopyStore(db)  # 模拟容器重启
    snap = s2.get(cid)
    assert snap["status"] == "finalized" and snap["revision"] == 2
    with pytest.raises(CopyFinalizedError):
        s2.claim(cid, "a", "alice")


def test_old_db_auto_migrates_to_status_open(tmp_path):
    # 手工构造无 status 列的旧库：既有副本默认 open、原 revision 不变
    db = tmp_path / "copies.db"
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
    conn.commit()
    conn.close()

    store = CopyStore(db)
    snap = store.get("c1")
    assert snap["status"] == "open"
    assert snap["revision"] == 5  # 原 revision 不变
    # 迁移后原行为与结稿均可用
    store.claim("c1", "a", "alice")
    store.release("c1", "a", "alice")
    done = store.finalize("c1", 7)
    assert done["status"] == "finalized" and done["revision"] == 8
    # 迁移幂等：再次打开不报错、不破坏数据
    again = CopyStore(db).get("c1")
    assert again["status"] == "finalized" and again["revision"] == 8


def test_failed_finalize_persists_nothing(store):
    cid = store.create(TEXT, [ann("a", 0, 3)])["id"]
    store.claim(cid, "a", "alice")
    with pytest.raises(CopyStillHeldError):
        store.finalize(cid, 2)
    with pytest.raises(RevisionMismatchError):
        store.finalize(cid, 1)
    snap = store.get(cid)
    assert snap["status"] == "open" and snap["revision"] == 2
    assert snap["annotations"][0]["holder"] == "alice"


# ---------- 并发：结稿与变更等价于某个先后次序 ----------

def test_concurrent_finalize_and_claim_exactly_one_ordering(store):
    cid = store.create(TEXT, [ann("a", 0, 3)])["id"]
    outcomes = {"claim": None, "finalize": None}
    barrier = threading.Barrier(2)

    def claim():
        barrier.wait()
        try:
            store.claim(cid, "a", "alice")
            outcomes["claim"] = "ok"
        except CopyFinalizedError:
            outcomes["claim"] = "finalized"

    def finalize():
        barrier.wait()
        try:
            store.finalize(cid, 1)
            outcomes["finalize"] = "ok"
        except (RevisionMismatchError, CopyStillHeldError):
            outcomes["finalize"] = "conflict"

    t1 = threading.Thread(target=claim)
    t2 = threading.Thread(target=finalize)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    snap = store.get(cid)
    assert snap["revision"] == 2  # 恰有一个变更成功
    if outcomes["finalize"] == "ok":
        # 结稿在前：领取必须失败，且无任何持有写入
        assert outcomes["claim"] == "finalized"
        assert snap["status"] == "finalized"
        assert snap["annotations"][0]["holder"] is None
    else:
        # 领取在前：结稿必须失败（版本过期或仍有持有者）
        assert outcomes["claim"] == "ok"
        assert outcomes["finalize"] == "conflict"
        assert snap["status"] == "open"
        assert snap["annotations"][0]["holder"] == "alice"


def test_concurrent_finalize_exactly_one_winner(store):
    cid = store.create(TEXT, [ann("a", 0, 3)])["id"]
    results = []
    barrier = threading.Barrier(8)

    def finalize():
        barrier.wait()
        try:
            store.finalize(cid, 1)
            results.append("ok")
        except (RevisionMismatchError, CopyFinalizedError):
            results.append("conflict")

    threads = [threading.Thread(target=finalize) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count("ok") == 1  # 恰有一个结稿成功
    snap = store.get(cid)
    assert snap["status"] == "finalized" and snap["revision"] == 2
