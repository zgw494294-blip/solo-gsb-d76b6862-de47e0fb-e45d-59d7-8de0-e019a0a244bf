"""特殊批注 id（空串、含 "/"、字面 "%2F"、含换行）的**真实 HTTP** 全流程回归。

FastAPI/Starlette 的 TestClient 会把请求路径百分号解码两次（httpx 构造 URL
时一次、TestClient 组装 ASGI scope 时 ``unquote`` 再一次），与 uvicorn 真实
服务端「只解码一次」的行为不一致，无法用它验证 "/" 与 "%2F" 的区分。这里
启动一个真实 uvicorn 服务端（临时端口、临时 SQLite 路径），经真实 HTTP
套接字完成 POST/GET/POST/POST/PATCH 全流程。

路径段按标准百分号编码构造（``urllib.parse.quote(id, safe="")``）：
- id "审阅/1" → "%E5%AE%A1%E9%98%85%2F1"（"/" 编码为 "%2F"）
- id "a%2Fb"  → "a%252Fb"（"%" 再编码为 "%25"，与 id "a/b" 区分）
- id ""       → ""（路径里表现为相邻的两个 "/"；PATCH 以 "/" 收尾）
- id "a\n"    → "a%0A"（换行编码为 "%0A"；服务端路由必须完整保留末尾换行，
  不得把 "a\n" 截成 "a"）
"""

import importlib
import os
import socket
import threading
import time
from urllib.parse import quote

import httpx
import pytest
import uvicorn


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("copies") / "copies.db"
    os.environ["COPY_DB_PATH"] = str(db_path)
    import app.main as main
    importlib.reload(main)  # 用本模块的临时数据库路径重新初始化全局仓库

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = uvicorn.Server(uvicorn.Config(
        main.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while time.time() < deadline:  # 等待服务端就绪
        try:
            httpx.get(f"http://127.0.0.1:{port}/", timeout=1)
            break
        except httpx.TransportError:
            time.sleep(0.05)
    else:
        raise RuntimeError("测试用 uvicorn 未能启动")

    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}",
                          timeout=5, follow_redirects=True) as c:
            yield c
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        main._store = None


def ann(id, start, end, label="L"):
    return {"id": id, "start": start, "end": end, "label": label}


def _create(client, anns, text="x" * 30):
    r = client.post("/api/copies", json={"text": text, "annotations": anns})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _seg(ann_id: str) -> str:
    return quote(ann_id, safe="")


def _url(cid, ann_id, action=None):
    """构造批注操作 URL；action 为 claim/release 或 None（PATCH）。"""
    url = f"/api/copies/{cid}/annotations/{_seg(ann_id)}"
    return url if action is None else f"{url}/{action}"


def _holders(snap):
    return {a["id"]: a["holder"] for a in snap["annotations"]}


# ---------- 创建 / 读取：特殊 id 原样存储与返回 ----------

def test_create_and_get_special_ids(client):
    special = ["", "审阅/1", "a%2Fb", "a b?c&d"]
    cid = _create(client, [ann(i, k, k + 1) for k, i in enumerate(special, start=1)])
    snap = client.get(f"/api/copies/{cid}").json()
    assert [a["id"] for a in snap["annotations"]] == special


# ---------- 空字符串 id：领取（整组）/ 冲突 / 修改 / 释放 ----------

def test_empty_id_full_flow(client):
    # 空串批注与 b 正长度重叠同组；c 独立
    cid = _create(client, [ann("", 0, 4), ann("b", 3, 7), ann("c", 12, 15)])

    r = client.post(_url(cid, "", "claim"), json={"reviewer": "alice"})
    assert r.status_code == 200, r.text
    assert _holders(r.json()) == {"": "alice", "b": "alice", "c": None}
    assert r.json()["revision"] == 2

    # 他人领取空串 id 所在整组 → 409，副本不变
    r = client.post(_url(cid, "", "claim"), json={"reviewer": "bob"})
    assert r.status_code == 409
    assert r.json()["detail"]["holder"] == "alice"
    snap = client.get(f"/api/copies/{cid}").json()
    assert snap["revision"] == 2

    # reviewer 非法 → 422，路由仍命中空串 id
    r = client.post(_url(cid, "", "claim"), json={"reviewer": ""})
    assert r.status_code == 422

    # 持有者 PATCH 空串 id：URL 以 / 收尾；缩小后与 b 拆开，b 保留原持有者
    r = client.patch(_url(cid, ""),
                     json={"reviewer": "alice", "start": 0, "end": 2,
                           "label": "改"})
    assert r.status_code == 200, r.text
    by_id = {a["id"]: a for a in r.json()["annotations"]}
    assert (by_id[""]["start"], by_id[""]["end"], by_id[""]["label"]) == \
        (0, 2, "改")
    assert [g["annotation_ids"] for g in r.json()["groups"]] == [[""], ["b"], ["c"]]

    # 释放空串批注（仅释放其所在组，即它自己）；b 已被拆开、仍由 alice 持有
    r = client.post(_url(cid, "", "release"), json={"reviewer": "alice"})
    assert r.status_code == 200, r.text
    assert _holders(r.json()) == {"": None, "b": "alice", "c": None}

    # 再次释放已无人持有的空串批注 → 409
    r = client.post(_url(cid, "", "release"), json={"reviewer": "alice"})
    assert r.status_code == 409
    snap = client.get(f"/api/copies/{cid}").json()
    assert _holders(snap) == {"": None, "b": "alice", "c": None}


# ---------- 含 "/" 的 id（审阅/1）：三项操作全流程 ----------

def test_slash_id_full_flow(client):
    cid = _create(client, [ann("审阅/1", 0, 4), ann("b", 3, 7),
                           ann("c", 12, 15)])

    # 路径段必须写作 %E5%AE%A1%E9%98%85%2F1；未编码的裸 "/" 同样命中
    assert _seg("审阅/1") == "%E5%AE%A1%E9%98%85%2F1"
    r = client.post(
        f"/api/copies/{cid}/annotations/审阅/1/claim",
        json={"reviewer": "alice"})  # httpx 发送时自动百分号编码
    assert r.status_code == 200, r.text
    assert _holders(r.json()) == {"审阅/1": "alice", "b": "alice", "c": None}

    r = client.post(_url(cid, "审阅/1", "claim"), json={"reviewer": "bob"})
    assert r.status_code == 409
    assert r.json()["detail"]["holder"] == "alice"

    r = client.patch(_url(cid, "审阅/1"),
                     json={"reviewer": "alice", "start": 0, "end": 4,
                           "label": "审改"})
    assert r.status_code == 200, r.text
    by_id = {a["id"]: a for a in r.json()["annotations"]}
    assert by_id["审阅/1"]["label"] == "审改"

    r = client.post(_url(cid, "审阅/1", "release"),
                    json={"reviewer": "alice"})
    assert r.status_code == 200, r.text
    assert all(h is None for h in _holders(r.json()).values())

    r = client.post(_url(cid, "审阅/1", "release"),
                    json={"reviewer": "alice"})
    assert r.status_code == 409  # 无人持有


# ---------- "/" 与 "%2F" 不得混同 ----------

def test_slash_and_literal_percent2f_are_distinct(client):
    cid = _create(client, [ann("a/b", 0, 2), ann("a%2Fb", 12, 14)])
    assert _seg("a/b") == "a%2Fb"
    assert _seg("a%2Fb") == "a%252Fb"

    r = client.post(_url(cid, "a/b", "claim"), json={"reviewer": "alice"})
    assert r.status_code == 200, r.text
    r = client.post(_url(cid, "a%2Fb", "claim"), json={"reviewer": "bob"})
    assert r.status_code == 200, r.text  # 另一条批注、另一个组
    snap = client.get(f"/api/copies/{cid}").json()
    assert _holders(snap) == {"a/b": "alice", "a%2Fb": "bob"}

    # 把 a/b 改到与 a%2Fb 正长度重叠 → 触及他人持有，409 且范围不变
    r = client.patch(_url(cid, "a/b"),
                     json={"reviewer": "alice", "start": 12, "end": 13,
                           "label": "X"})
    assert r.status_code == 409
    assert r.json()["detail"]["foreign"] == {"a%2Fb": "bob"}
    snap = client.get(f"/api/copies/{cid}").json()
    by_id = {a["id"]: a for a in snap["annotations"]}
    assert (by_id["a/b"]["start"], by_id["a/b"]["end"]) == (0, 2)
    assert by_id["a%2Fb"]["holder"] == "bob"

    # 各自的合法修改互不干扰
    r = client.patch(_url(cid, "a/b"),
                     json={"reviewer": "alice", "start": 0, "end": 1,
                           "label": "X"})
    assert r.status_code == 200
    r = client.patch(_url(cid, "a%2Fb"),
                     json={"reviewer": "bob", "start": 12, "end": 13,
                           "label": "Y"})
    assert r.status_code == 200
    # 释放必须精确命中各自的 id，不能串台
    assert client.post(_url(cid, "a/b", "release"),
                       json={"reviewer": "bob"}).status_code == 409
    assert client.post(_url(cid, "a%2Fb", "release"),
                       json={"reviewer": "alice"}).status_code == 409
    assert client.post(_url(cid, "a/b", "release"),
                       json={"reviewer": "alice"}).status_code == 200
    assert client.post(_url(cid, "a%2Fb", "release"),
                       json={"reviewer": "bob"}).status_code == 200


# ---------- 含其他保留字符的任意唯一字符串 id ----------

def test_id_with_reserved_chars(client):
    ann_id = "a b?c&d#e"
    cid = _create(client, [ann(ann_id, 0, 3)])
    assert client.post(_url(cid, ann_id, "claim"),
                       json={"reviewer": "alice"}).status_code == 200
    assert client.patch(_url(cid, ann_id),
                        json={"reviewer": "alice", "start": 0, "end": 2,
                              "label": "X"}).status_code == 200
    assert client.post(_url(cid, ann_id, "release"),
                       json={"reviewer": "alice"}).status_code == 200


# ---------- 404：特殊 id 或副本不存在，路由命中但资源缺失 ----------

def test_special_ids_unknown_resources_404(client):
    cid = _create(client, anns=[])
    assert client.post(_url(cid, "no/such", "claim"),
                       json={"reviewer": "alice"}).status_code == 404
    assert client.post(_url(cid, "", "release"),
                       json={"reviewer": "alice"}).status_code == 404
    assert client.patch(_url(cid, ""),
                        json={"reviewer": "alice", "start": 0, "end": 1,
                              "label": "X"}).status_code == 404
    # 不存在的副本 + 含斜杠批注 id：copy_id 段先判定 404
    assert client.post("/api/copies/deadbeef/annotations/a/b/claim",
                       json={"reviewer": "alice"}).status_code == 404


# ---------- 含换行的 id：创建 / 读取原样保留，不拒收、不改写 ----------

def test_newline_ids_create_and_get(client):
    special = ["a", "a\n", "a\nb", "\n", "行\n换", "a\n\n"]
    cid = _create(client, [ann(i, k, k + 1) for k, i in enumerate(special, start=1)])
    snap = client.get(f"/api/copies/{cid}").json()
    assert [a["id"] for a in snap["annotations"]] == special


# ---------- 末尾换行的 id（"a\n" → "%0A"）：领取 / 冲突 / 修改 / 释放 ----------

def test_trailing_newline_id_full_flow(client):
    # "a\n" 与 b 正长度重叠同组；c 独立
    cid = _create(client, [ann("a\n", 0, 4), ann("b", 3, 7), ann("c", 12, 15)])
    assert _seg("a\n") == "a%0A"

    r = client.post(_url(cid, "a\n", "claim"), json={"reviewer": "alice"})
    assert r.status_code == 200, r.text
    assert _holders(r.json()) == {"a\n": "alice", "b": "alice", "c": None}
    assert r.json()["revision"] == 2

    # 他人领取 "a\n" 所在整组 → 409，副本不变
    r = client.post(_url(cid, "a\n", "claim"), json={"reviewer": "bob"})
    assert r.status_code == 409
    assert r.json()["detail"]["holder"] == "alice"
    assert client.get(f"/api/copies/{cid}").json()["revision"] == 2

    # 持有者 PATCH "a\n"：缩小后与 b 拆开，b 保留原持有者
    r = client.patch(_url(cid, "a\n"),
                     json={"reviewer": "alice", "start": 0, "end": 2,
                           "label": "改"})
    assert r.status_code == 200, r.text
    by_id = {a["id"]: a for a in r.json()["annotations"]}
    assert (by_id["a\n"]["start"], by_id["a\n"]["end"],
            by_id["a\n"]["label"]) == (0, 2, "改")
    assert [g["annotation_ids"] for g in r.json()["groups"]] == \
        [["a\n"], ["b"], ["c"]]

    # 释放 "a\n"（仅其所在组，即它自己）；b 已被拆开、仍由 alice 持有
    r = client.post(_url(cid, "a\n", "release"), json={"reviewer": "alice"})
    assert r.status_code == 200, r.text
    assert _holders(r.json()) == {"a\n": None, "b": "alice", "c": None}

    # 再次释放已无人持有的 "a\n" → 409
    assert client.post(_url(cid, "a\n", "release"),
                       json={"reviewer": "alice"}).status_code == 409


# ---------- 换行在 id 中间（"a\nb" → "%0A"）：与 "a"、"a\n" 同副本各自定位 ----------

def test_newline_inside_id_coexists_with_prefixes(client):
    cid = _create(client, [ann("a", 0, 2), ann("a\n", 5, 7), ann("a\nb", 12, 15)])
    assert _seg("a\nb") == "a%0Ab"

    # 三个 id 互为前缀 / 子串，必须各自完整命中、互不串台
    assert client.post(_url(cid, "a\nb", "claim"),
                       json={"reviewer": "alice"}).status_code == 200
    assert client.post(_url(cid, "a\n", "claim"),
                       json={"reviewer": "bob"}).status_code == 200
    assert client.post(_url(cid, "a", "claim"),
                       json={"reviewer": "carol"}).status_code == 200
    snap = client.get(f"/api/copies/{cid}").json()
    assert _holders(snap) == {"a": "carol", "a\n": "bob", "a\nb": "alice"}

    # 各自的合法修改互不干扰
    r = client.patch(_url(cid, "a\nb"),
                     json={"reviewer": "alice", "start": 12, "end": 14,
                           "label": "AB"})
    assert r.status_code == 200, r.text
    r = client.patch(_url(cid, "a\n"),
                     json={"reviewer": "bob", "start": 5, "end": 6,
                           "label": "A换"})
    assert r.status_code == 200, r.text
    snap = client.get(f"/api/copies/{cid}").json()
    by_id = {a["id"]: a for a in snap["annotations"]}
    assert (by_id["a\nb"]["end"], by_id["a\nb"]["label"]) == (14, "AB")
    assert (by_id["a\n"]["end"], by_id["a\n"]["label"]) == (6, "A换")
    assert (by_id["a"]["start"], by_id["a"]["end"],
            by_id["a"]["label"]) == (0, 2, "L")

    # 释放必须精确命中各自的 id
    assert client.post(_url(cid, "a\nb", "release"),
                       json={"reviewer": "bob"}).status_code == 409
    assert client.post(_url(cid, "a\nb", "release"),
                       json={"reviewer": "alice"}).status_code == 200
    assert client.post(_url(cid, "a\n", "release"),
                       json={"reviewer": "bob"}).status_code == 200
    assert client.post(_url(cid, "a", "release"),
                       json={"reviewer": "carol"}).status_code == 200
    snap = client.get(f"/api/copies/{cid}").json()
    assert all(h is None for h in _holders(snap).values())


# ---------- 仅一个换行的 id（"\n" → "%0A"） ----------

def test_lone_newline_id_full_flow(client):
    cid = _create(client, [ann("\n", 0, 2), ann("b", 5, 7)])
    assert _seg("\n") == "%0A"

    assert client.post(_url(cid, "\n", "claim"),
                       json={"reviewer": "alice"}).status_code == 200
    snap = client.get(f"/api/copies/{cid}").json()
    assert _holders(snap) == {"\n": "alice", "b": None}

    assert client.patch(_url(cid, "\n"),
                        json={"reviewer": "alice", "start": 0, "end": 1,
                              "label": "换"}).status_code == 200
    assert client.post(_url(cid, "\n", "release"),
                       json={"reviewer": "alice"}).status_code == 200
    snap = client.get(f"/api/copies/{cid}").json()
    assert _holders(snap) == {"\n": None, "b": None}


# ---------- 回归：PATCH ".../a%0A" 不得把 "a\n" 当成 "a" 错改 ----------

def test_trailing_newline_patch_never_hits_prefix_id(client):
    cid = _create(client, [ann("a", 0, 2), ann("a\n", 5, 7)])

    # alice 只持有 "a"；对 "a\n"（未持有）的 PATCH 必须 409，
    # 且 "a" 与 "a\n" 的内容、持有权及 revision 均不变
    # （修复前路由会把末尾换行丢掉，静默改到 "a"）
    assert client.post(_url(cid, "a", "claim"),
                       json={"reviewer": "alice"}).status_code == 200
    rev = client.get(f"/api/copies/{cid}").json()["revision"]
    r = client.patch(_url(cid, "a\n"),
                     json={"reviewer": "alice", "start": 0, "end": 1,
                           "label": "错改"})
    assert r.status_code == 409
    snap = client.get(f"/api/copies/{cid}").json()
    assert snap["revision"] == rev
    by_id = {a["id"]: a for a in snap["annotations"]}
    assert (by_id["a"]["start"], by_id["a"]["end"],
            by_id["a"]["label"]) == (0, 2, "L")
    assert by_id["a"]["holder"] == "alice"
    assert (by_id["a\n"]["start"], by_id["a\n"]["end"],
            by_id["a\n"]["label"]) == (5, 7, "L")
    assert by_id["a\n"]["holder"] is None

    # "a\n" 的持有者 PATCH：命中 "a\n" 本身，"a" 不受影响
    assert client.post(_url(cid, "a\n", "claim"),
                       json={"reviewer": "bob"}).status_code == 200
    r = client.patch(_url(cid, "a\n"),
                     json={"reviewer": "bob", "start": 5, "end": 6,
                           "label": "换改"})
    assert r.status_code == 200, r.text
    by_id = {a["id"]: a for a in r.json()["annotations"]}
    assert (by_id["a\n"]["start"], by_id["a\n"]["end"],
            by_id["a\n"]["label"]) == (5, 6, "换改")
    assert (by_id["a"]["start"], by_id["a"]["end"],
            by_id["a"]["label"]) == (0, 2, "L")


# ---------- 换行 id 的 404：路由命中但批注不存在，副本不变 ----------

def test_newline_id_unknown_404(client):
    cid = _create(client, [ann("a", 0, 2)])
    assert client.post(_url(cid, "a\n", "claim"),
                       json={"reviewer": "alice"}).status_code == 404
    assert client.post(_url(cid, "a\n", "release"),
                       json={"reviewer": "alice"}).status_code == 404
    assert client.patch(_url(cid, "a\nb"),
                        json={"reviewer": "alice", "start": 0, "end": 1,
                              "label": "X"}).status_code == 404
    # 失败不改变内容、持有权与 revision
    snap = client.get(f"/api/copies/{cid}").json()
    assert snap["revision"] == 1
    assert _holders(snap) == {"a": None}
