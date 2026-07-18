# semantic-guess-solver — 中文版

> 基于语义相似度（BGE 嵌入 + 黑盒打分）与离线 Round-1 余弦排序器，
> 对 **xiaoce.fun GuessWord 每日挑战** 进行逆向工程。纯 numpy、NDJSON 回放格式、TDD 测试驱动。

[![tests](https://img.shields.io/badge/tests-52%20passed-brightgreen)]()
[![python](https://img.shields.io/badge/python-≥3.10-blue)]()
[![deps](https://img.shields.io/badge/runtime%20deps-numpy%20only-orange)]()
[![license](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## 项目简介

一个研究级的求解器，针对 [xiaoce.fun](https://xiaoce.fun) 上的 **GuessWord** 每日谜题。
游戏每轮展示 30 个候选词，并对每次猜测返回一个黑盒相似度分数（范围约 0.4–1.0）。
答案是一个 2 字中文词语。

核心算法原理可以概括为：**语义嵌入空间 + 质心法启发式搜索 + 核岭回归（KRR）黑盒优化**。

以下是具体的步骤和机制解析：

### 1. 候选词与语义嵌入初始化（Preparation）

* **词库建设**：项目内置了一个包含 2257 个双字中文词的候选词库 (`data/cand_words.json`)。
* **向量化（Embedding）**：利用 **BGE 中文文本嵌入模型** (`BGE-base-zh`)，预先将这 2257 个候选词转化为 768 维的稠密向量（float32，并进行了单位化 Norm 变换）。这样，词语之间的“字面含义相近程度”就转化为了高维空间中的“几何距离”。

### 2. 探索与搜索策略（Search & Exploration）

在游戏开始后，求解器通过循环发送探测词（Probe）来逼近目标词：

* **前 30 探（质心法 / Centroid）**：
* 游戏初期，由于拿到的样本很少，求解器首先使用类似于“质心法”的雏形算法进行广度探测。
* 它通过观察前几轮猜测返回的分数，在 768 维空间中估算正确答案的大致中心位置，并筛选出距离该中心最近的下一个高潜力词汇发送给 API。


* **动态聚类识别（Cluster Type Detection）**：
* 求解器会根据前 30 探的反馈判断当前词汇团簇的特征：如果最高得分 $>0.7$，说明是**密集团簇（Dense）**，通常对应城市、食物、具体地名等；如果最高得分 $<0.5$，说明是**稀疏团簇（Sparse）**，通常对应抽象词。



### 3. 核岭回归黑盒拟合（KRR Optimization - 破局关键）

当遇到高分瓶颈（例如在 0.82 附近陷入局部最优平台期）或识别为稀疏抽象词时，算法会自动触发 **核岭回归（Kernel Ridge Regression, KRR）** 预测器：

* **黑盒建模**：xiaoce API 的评分机制对求解器而言是一个黑盒。KRR 的任务就是利用已猜过的词（自变量：BGE 向量）和 API 返回的分数（因变量：0~1 之间的得分），在线构建一个非线性的数学拟合表面。
* **闭式解加速**：KRR 拥有闭式解（Closed-form solution），无需复杂的深度学习迭代训练，纯靠 `numpy` 矩阵运算就能在毫秒级内完成 5×5 级别的网格搜索和超参数拟合。
* **主动学习（Active Learning）**：拟合出评分表面后，KRR 会预测剩余候选词库中哪个词的得分可能最高（即寻找函数极大值点），然后将其作为下一个探测词发给 API。拿到新分数后，再把数据喂回 KRR 更新模型，重复这一主动学习循环。

### 4. 辅助工程优化

除了核心数学算法，项目在工程上也做了一些保证求解高效率的闭环：

* **令牌桶限速（TokenBucket）**：内置 `ratelimit.py`，严格控制请求速率在 0.8 tokens/s 左右，防止被 API 封禁。
* **NDJSON 重放与回归测试**：将每次破解的过程以 NDJSON 格式记录下来（如 `case11_replay.ndjson`），方便离线进行算法模型的对比和回归测试（`replay_diff.py`）。

它并不是盲目地去字典里撞运气，而是把猜词游戏变成了一个“在高维空间中寻找未知目标点”的数学优化问题。先用质心法粗定位，再用核岭回归（KRR）去精细拟合 API 的评分曲线，从而实现用极少的探测次数（甚至个位数）直接抓到最终答案。
本库交付 **Round 1（离线排序器）+ Round 2（在线探测）**：


| 轮次 | 模块 | 是否联网 |
| --- | --- | --- |
| **Round 1** — 纯 numpy 余弦排序器 | `sgs.replay`, `sgs.rank`, `sgs.round1` | 否 |
| **Round 2** — 预言机契约 + 批量探测 + 限速 | `sgs.oracle`, `sgs.ratelimit`, `sgs.probe` | 是（或使用 fake） |

浏览器线协议适配层（Playwright/Chromium CDP）与主动学习循环位于 **Round 3+** —— 见 [Roadmap](#roadmap)。

---

## Round 1 为何有效

针对 5 个有据可查的谜题（案例 1–5，参见 [`references/`](references/)）的经验假设：

1. **BGE-zh-base 768 维嵌入** 在余弦空间中使语义相近的中文 token 聚类。
2. **预言机分数与到未知答案的余弦相似度近似单调**。
3. **即便只有 3–5 条带噪观测**，也能将一个分数加权中心点拉向正确的语义簇——这就是主动学习的信号。

在 case-5 上，2 阶段聚类收窄 + 1 次日终探针的策略，用 100 词种子 → 20 词 pivot → 单词答案的路径把分数打到 `0.989`（答案"萧山"）。

---

## 安装

```bash
pip install -e .[dev]
```

测试覆盖 Python 3.10 – 3.12。**运行时依赖：仅 numpy。**

---

## 快速上手

```bash
# 1. 用若干词探测预言机（浏览器侧；不在本库范围内）。
# 2. 将观测记录为 NDJSON —— 每行一条记录：
cat > replay/376634286041.ndjson <<'EOF'
{"word": "剑客", "score": 0.398, "ts": "2026-07-14T07:55:00Z"}
{"word": "武士", "score": 0.481, "ts": "2026-07-14T07:55:08Z"}
{"word": "忍者", "score": 0.612, "ts": "2026-07-14T07:55:17Z"}
{"word": "浪人", "score": 0.527, "ts": "2026-07-14T07:55:25Z"}
EOF

# 3. 运行 Round 1 —— 获取接下来要探测的 30 个词。
python -m sgs.round1 \
    --replay     replay/376634286041.ndjson \
    --candidates /path/to/cand_words.json \
    --embeddings /path/to/cand_emb.npy \
    --batch-size 30 \
    --out        replay/376634286041-next.ndjson
```

CLI 逐行输出 `rank  word  cosine`，并可选择性地将 `{"word", "rank", "score"}` 记录写入 NDJSON 文件。

---

## API

### `sgs.replay` — NDJSON 回放 I/O + sha256 指纹

```python
from sgs.replay import write_replay, read_replay, stream_replay, fingerprint

write_replay(Path("obs.ndjson"), [
    {"word": "忍者", "score": 0.989, "ts": "2026-07-14T08:11:32Z",
     "correct": True, "doubleScore": False},
])
records = read_replay(Path("obs.ndjson"))            # list[dict]
for rec in stream_replay(Path("big.ndjson")): ...    # 内存友好
sha = fingerprint(Path("obs.ndjson"))                # hex sha256
```

必需键：`word`、`score`、`ts`。
可选键按原样保留：`correct`、`doubleScore` 等。

### `sgs.rank` — 嵌入中心点 + 余弦排序

```python
from sgs.rank import load_corpus, fit_centroid, rank

words, emb = load_corpus("cand_words.json", "cand_emb.npy")
top30 = rank(
    observations=[("忍者", 0.612), ("剑客", 0.398), ("武士", 0.481)],
    words=words,
    emb=emb,
    top_k=30,                    # 默认 30
    exclude_observed=True,       # 设为 False 可审计答案的排名
)
# top30 == [(word, cosine), ...] 按降序排列
```

### `sgs.round1` — CLI

```text
usage: python -m sgs.round1 [-h] --replay REPLAY --candidates CANDIDATES
                            --embeddings EMBEDDINGS [--batch-size 30]
                            [--out OUT] [--include-correct]
```

---

## 开发

### 运行测试

```bash
pytest                       # 27 个测试，约 0.7s，零网络
```

### 目录结构

```text
sgs/
  __init__.py     # __version__
  replay.py       # NDJSON + sha256
  rank.py         # 中心点 + 余弦
  round1.py       # CLI
tests/
  test_replay.py  # 10 个测试
  test_rank.py    # 12 个测试
  test_round1.py  # 5 个子进程测试
```

### 设计原则

1. **Round 1 是纯 numpy** —— 零业务逻辑、零网络代码、不依赖浏览器探测（Round 2）。
2. **TDD**：测试先行；失败驱动 API 表面的演进（例如 `exclude_observed: bool` 取代了一个笨拙的 `exclude: set` 参数）。
3. **NDJSON + sha256** —— 回放文件可防篡改，并能以流式方式处理超长会话。
4. **无静默回退** —— 任何语料或分数上的错误都立即抛出。

---

## 注意事项

- 本库**不会**主动探测预言机。请与 Playwright/浏览器抓取脚本（Round 2，WIP）搭配使用。
- BGE 嵌入体积较大（38930 个中文词约 120 MB）。**不**入库；请单独下载或基于 [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) 生成。
- "答案为 2 字"这一约束**不在数学层面强制**——可传入任意长度的语料。案例 1–5 恰好都是 2 字答案；API 本身与长度无关。

---

## Roadmap

| 轮次 | 状态 | 内容 |
| --- | --- | --- |
| **1. 离线排序器** | ✅ `v0.1.0` | numpy 余弦 + NDJSON 回放 + sha256（27 个测试） |
| **2. 在线探测** | ✅ `v0.2.0` | `Oracle` 协议 + `TokenBucket` + 批量探测 + 命中即停（52 个测试） |
| **3. 浏览器线协议** | 计划中 | Playwright/Chromium CDP、持久上下文、一次性人工登录、`fcntl.flock` 防双开 |
| **4. 主动学习** | 计划中 | `U = α·pred + β·uncert + γ·diversity`，多轮收敛 |
| **5. 端到端** | 计划中 | `dry-run` / `assisted` / `supervised` / `live` 模式；闸门：`--max-probes`、`--max-domain-switches`、`--stop-on-plateau` |
| **6. 回放回归** | 计划中 | NDJSON 驱动的离线回归，含 golden diff |

---

## 许可证

MIT。参见 [`LICENSE`](LICENSE)。

---

## 参考资料

- 案例研究（1–5 谜题记录）：`references/xiaoce-fun-case-study.md`
- 配套 skill（位于 Hermes Agent 内）：`xiaoce-fun-case-study-375865943437`