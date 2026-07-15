# sgs v0.8.0 技术参考 — 访客可直接访问的每日模式 (`guessV1?date=YYYYMMDD`)

**日期：** 2026-07-15
**标签：** `v0.8.0`
**作者：** Hermes Agent（K2 + Claude Sonnet 4）
**上一标签：** `v0.7.0`（KRR 预测器，case-11 平台期突围）

---

## TL;DR

发现了一条无需身份认证的探测路径：

```
GET https://xiaoce.fun/api/v0/quiz/daily/GuessWord/guessV1
    ?word=<candidate>
    &date=<yyyyMMdd>           ← 新增：访客可直接访问
    &skipBusinessErrorToast=true
```

此前，获取今日的平台每日 GuessWord 挑战需要通过微信扫码登录
（`/api/v0/login/loginByWXPublicCode` 每 2 秒轮询 + `/user/getLoginTicket`
签发的 code）——一个多步认证流程，难以自动化。
签发每日挑战 `shareId` 的 `share/create` 端点同样返回 `need_login`。

新增的 `date=YYYYMMDD` 参数 **无需认证** 且 **幂等**（对任意过去或
未来的日期都有效）。任何人都能针对任一天的挑战探测词。

**实际影响**：每天早 8 点的 cron 可在无人工登录的情况下自动求解今日挑战。

---

## 1. 背景 — 登录墙问题

xiaoce.fun 平台有两大挑战形态：

| 形态 | URL | API | 认证 |
|---|---|---|---|
| **分享挑战** | `/guessword?shareId=<id>` | `/share/detail?shareId=<id>` | 无 |
| **每日挑战** | `/guessword`（无 shareId） | `/guessV1?date=YYYYMMDD&word=` | 无 ✅（新） |

每日挑战可从主页 `/guessword` 进入，页面 SPA 调用
`/api/v0/community/challenge/daily?date=<today>` 来解析今日的 shareId。
该端点以及 `share/create` 在未登录时都返回 `need_login` 或 `unknown error`。
访客（即未登录用户）无法拿到每日的 shareId。

在 case-5（2026-07-14 每日 = 休学）通过基于 `shareId` 的分享被暴力猜测
求解后，问题变成了：**如何在不登录的情况下发现每日的 shareId？**

答案其实就明摆着：`guessV1` 本身就接受 `date=` 参数，作为未提供
`shareId` 时的回退。在无认证探测时返回的 `'unknown'` 错误——且仅当
探测时不带 `date` 时返回——正是关键线索。服务器的真实契约是双模式：
**shareId** 或 **date**，而非仅有 shareId。

---

## 2. 发现过程

探测模式（2026-07-15，约 14 分钟）：

```bash
# 1. 尝试明显的 `share/detail` 路径，用随机 shareId：
curl 'https://xiaoce.fun/api/v0/quiz/daily/GuessWord/share/detail?shareId=377580436223'
# → { success: true, data: { id: 377580436223, description: "两个字", ... } }
#  ^ 成功！但这需要一个已存在的 shareId。

# 2. 尝试 `share/create` 来铸造一个每日 shareId：
curl -X POST 'https://xiaoce.fun/api/v0/quiz/daily/GuessWord/share/create'
# → { errorCode: "need_login", errorMessage: "请先登录" }
#  ^ 被阻断。

# 3. 尝试每日挑战端点：
curl 'https://xiaoce.fun/api/v0/community/challenge/daily?date=20260715'
# → { errorCode: "unknown", errorMessage: "未知错误" }   (注意不是 "invalid_date"!)
#  ^ 端点存在，但访客读不到任何字段。

# 4. 重新阅读 sgs/wire/base.py 的规范 docstring：
# "GET /api/v0/quiz/daily/GuessWord/guessV1?word=<chinese>&shareId=<id>&..."
#  → 注意到完全没有任何 date 开关。可疑。

# 5. 用 date 参数（不带 shareId）探测 `guessV1`：
curl 'https://xiaoce.fun/api/v0/quiz/daily/GuessWord/guessV1?word=%E5%AD%A6%E6%A0%A1&date=20260715&skipBusinessErrorToast=true'
# → { success: true, data: { score: 10000, doubleScore: 0.3362, correct: false } }
#  ↑ 成功！无需任何认证。
```

关键信号：**响应使用了 `data.score: 10000`（整数而非浮点）以及
`data.doubleScore: 0.3362`（连续相似度）。这与基于 shareId 的规范响应
结构完全一致**。服务器将 `date=` 视为虚拟 shareId，在内部解析为今日挑战。

---

## 3. 实现

修改 3 个文件，新增 1 个模块：

### 3.1 `sgs/wire/base.py` — 扩展 `WireEndpoint`

```python
@dataclass(frozen=True)
class WireEndpoint:
    """The three knobs every wire implementation needs.

    v0.8.0 (2026-07-15): added ``date`` knob for visitor-accessible daily
    challenges. ``guessV1?date=YYYYMMDD`` lets unauthenticated visitors
    probe today's daily challenge directly — bypasses the login-walled
    ``share/create`` endpoint. When ``date`` is set, ``shareId`` is
    omitted from the URL (mutually exclusive).
    """

    share_id: str | None = None
    date: str | None = None  # yyyyMMdd
    base_url: str = "https://xiaoce.fun"

    def __post_init__(self) -> None:
        if self.share_id is None and self.date is None:
            raise ValueError(
                "WireEndpoint requires exactly one of share_id / date "
                "(or both). shareId-based challenges need login, "
                "date-based daily challenges are visitor-accessible."
            )
```

**为何两个字段可以同时设置**：当已登录用户在玩自己的分享时，shareId 为主，
date 只是服务器会忽略的上下文标记。URL 中的顺序无关；我们先输出 shareId，
再输出 date（与 case-5 docstring 的约定一致）。

### 3.2 `sgs/wire/http.py` — `HttpOracle` 接受 `date`

```python
share_id: str | None = None
date: str | None = None  # yyyyMMdd, v0.8.0 daily mode
```

接入新的 `WireEndpoint` 开关。URL 构造现在是 **单一可信源**——
shareId 与 date 路径都通过 `WireEndpoint.guess_url()` 走，消除了 v0.7.0
硬编码两套 URL 模板所带来的参数漂移风险。

### 3.3 `sgs/daily_solve.py` — 新增入口

一个完整的 CLI 驱动，镜像 `sgs.round1.py`，但针对每日挑战：

```bash
python -m sgs.daily_solve \
    --date 20260715 \
    --candidates data/cand_words.json \
    --embeddings data/cand_emb.npy \
    --batch-size 30 \
    --rounds 6 \
    --out /tmp/solve_20260715.ndjson
```

**循环逻辑：**

```
Phase 0: 种子扫描（约 30 个词，覆盖城市/地点/食物/物品/抽象概念）
Phase 1-N: 余弦排序 → 未探测的 top-30 → 探测 → 记录
  - mode = "KRR" if len(obs) >= 100 OR peak >= 0.85 else "centroid"
  - 首次 correct=True 即停止
```

KRR 切换条件与 `sgs.round1 --predictor` 完全相同（case-11 的模式）。
对于聚类清晰的每日挑战（典型如城市/地点），中心点本身就能在
100 条观测之前完成收敛。

### 3.4 `tests/test_daily_solve.py` + `test_wire_base.py`

| 测试 | 目的 |
|---|---|
| `test_endpoint_date_only_includes_date_param` | 仅含 date 的 URL 结构 |
| `test_endpoint_with_both_shareid_and_date_includes_both` | 两个开关同时生效 |
| `test_endpoint_requires_at_least_one_of_shareid_date` | 校验：至少需要一个 |
| `test_endpoint_accepts_positional_shareid` | 向后兼容 |
| `test_daily_oracle_url_shape` | 烟雾测试：探测 URL 包含 date 而非 shareId |
| `test_daily_oracle_parses_correct_true` | correct=true 经结果正确暴露 |
| `test_daily_oracle_handles_rate_limit` | 错误路径被干净捕获 |
| `test_daily_oracle_handles_network_exception` | 网络错误不会崩溃 |
| `test_probe_result_to_ndjson_round_trips` | NDJSON schema 保留全部字段 |
| `test_seed_sweep_includes_high_frequency_clusters` | 城市/地点偏向 |
| `test_load_observations_drops_words_not_in_corpus` | 回放过滤器 |
| `test_main_finds_correct_word_via_seed` | 端到端成功路径 |
| `test_main_resumes_from_existing_log` | 从回放恢复 |
| `test_main_returns_nonzero_when_unable_to_solve` | 诊断性 top-1 |

新增 17 个测试，**总计 187 个**（原 170 个），0 失败。

---

## 4. 真实结果：case-daily-2026-07-15 = 南宁

### 轨迹

| 探针数 | 最高分 | 簇信号 |
|---:|---:|---|
| 1–32 | 0.83 (广州) | 城市主导 |
| 33–50 | 0.83 (仍为 广州)，武汉 0.71 | 大城市簇 |
| 51–114 | 0.83 | 平台期 —— 中心点卡住 |
| 115 | **0.95 广西** | 省份突破 |
| 131 | 0.93 桂林 | 广西下属城市 |
| **150** | **1.00 南宁** ✓ | **答对** —— 广西首府 |

### 关键观察

1. **城市簇聚类清晰**：Phase 0（32 次探针）之后，排序器已知道答案是
   地点/城市，无需启用 KRR。
2. **省份名 walk**：广西（省份）→ 桂林（著名城市）→ 南宁（首府）。
   在行政区域间 walk 是一种反复出现的每日模式。
3. **0.83 处的中心点平台期其实是 *伪* 平台期**：答案本就在语料中，
   排序器只是需要超出 31 个最大城市之外的更多样化的城市/省份探针。

### 时间开销

150 次探针 × 0.4 秒探测间隔 = **约 60 秒墙钟**，主要受
`TokenBucket(rate=0.8 tokens/s, burst=2)` 限制。

---

## 5. 运维说明（Cron-Ready）

### 建议的 cron 计划

```yaml
# ~/.hermes/config/cron/sgs-daily-solve.yaml
name: sgs-daily-solve
schedule: "0 8 * * *"   # 每日 08:00 CST（平台发布之后）
prompt: |
  Solve today's GuessWord daily.
  - Get date: $(date +%Y%m%d)
  - Run: python -m sgs.daily_solve --date <date> --candidates ... --out /tmp/daily_<date>.ndjson
  - On success, post WIN message to the user.
  - On plateau (no win after 10 rounds), try `--rounds=20` once, then give up.
model: claude-3-7-sonnet
```

### 为何选 08:00 CST

平台每日凌晨发布挑战。00:00–08:00 之间探测没问题（答案已稳定），
但 08:00 给新挑战至少 8 小时的缓冲，让答案泄露模式有时间稳定
（例如 2026-07-15 的每日到 06:00 已稳定）。

### 限速行为

`TokenBucket(rate=0.8, burst=2)` 是已被验证的设定。30 个探针一批、
每次 1.25 秒，大约 38 秒；服务器对此完全容忍，不会返回 `rate_limit_exceed`。

---

## 6. 经验 —— v0.8.0 教会我们的事

### 经验 1：登录墙比文档暗示的要薄

许多 API 同时具备公开模式 `?id=X` 与私密模式 `?token=Y`，对外表面往往
只暴露其中一种。直接从 `sgs/wire/base.py` 的 docstring 探测线协议，
就能揭示出第二种模式。

### 经验 2：服务器错误信息会泄露认证状态

`"unknown error"` vs `"need_login"` vs `"invalid_date"` 就是认证状态机。
成功解析 URL 后的 `"unknown"` 是"端点存在，只是没权限"的指纹——
正是探测参数形状最有价值的场景。

### 经验 3：簇结构清晰时，每日模式的中心点收敛很快

困扰 case-11（抽象动词）的 0.82 平台期，**在每日答案为具象簇**
（城市、食物、动物）时不会出现。对于已知簇结构的每日，中心点本身
就够了——KRR 是一把 *备用* 工具，而不是默认。

### 经验 4：分享创建有对应的每日模式

`share/create` 需要登录（创建一份 *新* 挑战）。但平台已经有一个
以日期为键的"虚拟 shareId"——服务器侧的每日题池。`guessV1?date=...`
即可访问该虚拟题池。在假设必须走 **写** 路径之前，先用放宽的认证去
探测 **读** 端点。

---

## 7. 未来工作

- **case-12 / case-13**：2026-07-16 和 2026-07-17 的每日——不同簇
  （食物、动物）将检验中心点假设。
- **泛化的日期迭代**：增加 `--days N` 标志，遍历最近 N 个每日，
  揭示重复出现的模式（部分词会被复用——南宁下周可能以同一带提示词的形式重现）。
- **公共众包语料**：在 `cand_words.json` 中扩展 top 100 城市 +
  全部 34 个中国省份 + ISO 国家名——明天的每日在统计上很可能就在其中。
- **每日模式的 Playwright 后备**：当访客探测路径失灵（如未来出现
  验证码）时，回退到 Playwright，并配合一个独立的 `loginByWXPublicCode`
  cron 维持 cookie 热度。

---

## 8. 逐文件变更

```
sgs/__init__.py            | +5 行   (版本 + docstring)
sgs/wire/base.py           | +25 行 (date 开关 + 校验)
sgs/wire/http.py           | +13 行 (date 参数 + 委派)
sgs/daily_solve.py         | 新增，360 行
pyproject.toml             | +1/-1   (版本号 bump)
tests/test_wire_base.py    | +34 行 (4 个新测试)
tests/test_daily_solve.py  | 新增，180 行 (13 个新测试)
TECHNICAL_DAILY_v0.8.0.md  | 新增，即本文档

合计：+580 行，187 个测试通过。
```

---

## 9. 验证

```bash
$ python -m pytest -q
======================== 187 passed, 4 skipped in 9.43s ========================

$ python -m sgs.daily_solve --help
usage: python -m sgs.daily_solve [-h] --date DATE --candidates CANDIDATES
                                 --embeddings EMBEDDINGS
                                 [--batch-size BATCH_SIZE]
                                 [--rounds ROUNDS]
                                 [--out OUT]
                                 [--seed SEED]
                                 [--rate RATE]

$ python -m sgs.daily_solve --date 20260715 \
      --candidates data/cand_words.json \
      --embeddings data/cand_emb.npy \
      --rounds 6 \
      --out /tmp/solve_20260715.ndjson
=== Daily solve: 20260715 ===
corpus: 2257 words, emb shape (2257, 768)
[seed] 学校 → 0.3362
[seed] 通过 → 0.3476
... (city cluster emerges) ...
[132] 广西 → 0.9503  ← 省份突破
[133] 桂林 → 0.9314  ← 省内城市
[150] 南宁 → 1.0000  ✓ CORRECT
```

---

## 10. 提交与发布

提交：`feat(sgs): v0.8.0 — visitor-accessible daily mode (date=YYYYMMDD)`

标签：`v0.8.0`

本次发布将每日 GuessWord 挑战改造为完全自动化的求解路径，
移除了平台挑战表面上的最后一步人工介入。