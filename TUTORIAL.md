# sgs 完整使用教程 — 从零开始到 daily 自动解

> 目标读者: 想玩 **xiaoce.fun GuessWord** 每日猜词游戏, 或者想用 **sgs 求解器** 自动化解题的人
> 阅读时间: 15 分钟
> 实战: 30 分钟

---

## 目录

- [这是给谁的](#这是给谁的)
- [5 分钟懂项目](#5-分钟懂项目)
- [环境准备](#环境准备)
- [教程 A: 手动玩一次](#教程-a-手动玩一次)
- [教程 B: 自动化 daily challenge](#教程-b-自动化-daily-challenge)
- [教程 C: 解 shareId 分享题](#教程-c-解-shareid-分享题)
- [教程 D: 离线回放与 KRR 训练](#教程 d-离线回放与 krr-训练)
- [教程 E: 自己造候选词库](#教程-e-自己造候选词库)
- [常见问题](#常见问题)
- [下一步](#下一步)

---

## 这是给谁的

如果你属于以下任意一类, 这教程对你有用:

| 你是 | 想做 | 走哪个教程 |
|---|---|---|
| 玩家 | "今天 daily 出了, 帮我解" | [教程 B](#教程-b-自动化-daily-challenge) |
| 玩家 | "朋友给我 shareId 链接, 帮我解" | [教程 C](#教程-c-解-shareid-分享题) |
| 工程师 | "这工具怎么装怎么跑" | [教程 A](#教程-a-手动玩一次) |
| 工程师 | "我想接 cron 每天 8 点自动跑" | [教程 B + cron 配法](#教程-b-自动化-daily-challenge) |
| ML 工程师 | "我想训练自己的 KRR" | [教程 D](#教程-d-离线回放与-krr-训练) |
| 数据工程师 | "我想换自己的词库" | [教程 E](#教程-e-自己造候选词库) |

---

## 5 分钟懂项目

**sgs (semantic-guess-solver)** 是一个针对 xiaoce.fun GuessWord 猜词游戏的自动化求解器.

**猜词游戏是什么**: xiaoce.fun 每天出一道 2 字中文词, 你猜一个它给你一个分数 0-1, **1 = 答对**.

**sgs 怎么猜的**:

1. 准备 2257 个候选 2 字词的 BGE 中文嵌入 (768 维向量, 已存到 `data/cand_emb.npy`)
2. 拿历史猜测 + 分数, 用**质心法** (centroid) 或 **KRR** (核岭回归) 算出"最值得猜的下一个词"
3. 把这个词发到 xiaoce API, 拿分数, 重复

**v0.8.0 的关键发现**: xiaoce 的 `guessV1?date=YYYYMMDD` 接口**免登录**, daily challenge 现在能完全 cron 自动化.

---

## 环境准备

### 系统要求

- **Python ≥ 3.10** (旧版可能也行, 我用 3.12 测过)
- **操作系统**: Linux / macOS / WSL 都行
- **网络**: 能访问 xiaoce.fun 即可 (无需 VPN, 国内直连)
- **磁盘**: 大概 10MB (主要是 `data/cand_emb.npy` 占 6.7MB)

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/arckalsun/semantic-guess-solver.git
cd semantic-guess-solver

# 2. 安装依赖 (只有 numpy)
pip install numpy
# 或者
pip install -r requirements.txt  # 如果之后补了 requirements 的话

# 3. 跑测试确认安装好
python -m pytest -q
# 期望输出: 187 passed, 4 skipped in ~9s
```

### 验证 (3 步 sanity check)

```bash
# 1. corpus 词数 = 2257
python3 -c "import json; print(len(json.load(open('data/cand_words.json'))))"

# 2. embed 形状 = (2257, 768)
python3 -c "import numpy as np; print(np.load('data/cand_emb.npy').shape)"

# 3. CLI 帮助
python -m sgs.daily_solve --help
# 期望输出: usage 显示所有 flag (--date, --candidates, --embeddings, --rounds, --out ...)
```

如果这 3 步都通过, 你就准备好跑了.

---

## 教程 A: 手动玩一次

> 5 分钟. 不写一行代码, 只跑命令. 看完你能理解整个项目怎么用.

### 场景

你想看看 sgs 怎么解 daily challenge, 单纯验证它能跑. **不需要 WeChat 扫码, 不需要登录**.

### 步骤

```bash
# 跑今天的 daily (8 月 1 日就写 20260801)
python -m sgs.daily_solve \
  --date 20260801 \
  --candidates data/cand_words.json \
  --embeddings data/cand_emb.npy \
  --batch-size 30 --rounds 6 \
  --out /tmp/solve_20260801.ndjson
```

### 你会看到什么

```
=== Daily solve: 20260801 ===
corpus: 2257 words, emb shape (2257, 768)
[  1] 学校 → 0.3362
[  2] 通过 → 0.3476
...
[  9]  完毕 → 0.5417
...
[ 47]  城市 → 0.7135  ← 触发 KRR (peak>0.85 + obs>100)
[ 48]  小时 → 0.9376  ← KRR 突破!
...
[150]  分钟 → 0.9890 ✓ CORRECT
```

**大约 1-7 分钟** (具体看 cluster 类型):
- **密集 cluster** (城市/食物/地名) → centroid 单跑, 60-150 探, 1 分钟左右
- **稀疏 cluster** (抽象名词/动词) → centroid 卡 0.5-0.6, 切 KRR, 5-15 分钟

### 检查结果

```bash
# 1. 简单看正确答案
python3 -c "
import json
correct = [json.loads(l) for l in open('/tmp/solve_20260801.ndjson') if json.loads(l).get('correct')]
print(correct[0] if correct else 'not solved yet')
"

# 2. 看分数轨迹
python3 -c "
import json
obs = [(json.loads(l)['word'], json.loads(l)['score']) 
       for l in open('/tmp/solve_20260801.ndjson') if json.loads(l).get('score')]
obs.sort(key=lambda x: -x[1])
print('Top 5 命中前 5 探:')
for w, s in obs[:5]:
    print(f'  {w} {s:.4f}')
"
```

### 何时需要重跑

| 情况 | 怎么判断 | 怎么办 |
|---|---|---|
| 没找到正确答案 | output 末行写 "退出无正确答案" | 加 `--rounds 12` 再试 |
| 跑得太久 | 超过 15 分钟还卡在 KRR | 用 `--rounds 4` 提前结束, 看 top-3 候选 |
| 想快 | 默认 `rate=0.8 tokens/s` | 加 `--rate 2.0` (但有 rate-limit 风险) |

---

## 教程 B: 自动化 daily challenge

> 10 分钟. 给不想每天手动跑命令的人.

### 场景

你不想每天起床想"今天 daily 出了, 跑一下", 想**自动跑**. sgs v0.8.0 的访客模式 API 让你能**完全 cron 化**这件事, 不用登录任何东西.

### 一次性配置 (5 分钟)

```bash
# 1. 准备一个目录 (建议放项目内)
mkdir -p /home/youruser/sgs-daily
cd /home/youruser/sgs-daily

# 2. 软链 sgs 的核心文件 (避免复制)
ln -s /path/to/semantic-guess-solver/sgs .
ln -s /path/to/semantic-guess-solver/data .

# 3. 写一个 run.sh (复用)
cat > run.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d)
LOG=/home/youruser/sgs-daily/solve_${DATE}.ndjson

python -m sgs.daily_solve \
  --date ${DATE} \
  --candidates data/cand_words.json \
  --embeddings data/cand_emb.npy \
  --batch-size 30 --rounds 8 \
  --out ${LOG} 2>&1

# 检查是否解出
if python3 -c "import json,sys; sys.exit(0 if any(json.loads(l).get('correct') for l in open('${LOG}')) else 1)"; then
  echo "✓ ${DATE} solved!"
  python3 -c "import json; print([json.loads(l) for l in open('${LOG}') if json.loads(l).get('correct')][0])"
else
  echo "✗ ${DATE} not solved, top-3:"
  python3 -c "import json; obs=sorted([(json.loads(l)['word'], json.loads(l)['score']) for l in open('${LOG}') if json.loads(l).get('score')], key=lambda x:-x[1])[:3]; print(obs)"
fi
EOF
chmod +x run.sh
```

### 配置 crontab

```bash
# 编辑 crontab
crontab -e

# 早上 8 点跑 (daily 0 点放出, 8 点后跑数据稳定)
0 8 * * * /home/youruser/sgs-daily/run.sh >> /home/youruser/sgs-daily/cron.log 2>&1
```

### 验证 (手工跑一次, 模拟 cron)

```bash
# 立刻跑一次 (用今天日期, 即使还没到 8 点)
cd /home/youruser/sgs-daily
bash run.sh

# 期望: "✓ YYYYMMDD solved!" + 答案详情
```

### 日常使用

完全不需要动手. 每天早上你会看到 `cron.log` 出现新的一行. 想查历史答案:

```bash
# 列所有 daily 的答案
for f in solve_2026*.ndjson; do
  date=$(basename $f .ndjson | sed 's/solve_//')
  answer=$(python3 -c "import json; print([json.loads(l)['word'] for l in open('$f') if json.loads(l).get('correct')][0])" 2>/dev/null || echo "未解出")
  echo "$date: $answer"
done
```

---

## 教程 C: 解 shareId 分享题

> 5 分钟. 处理朋友分享给你的题 (不是 daily).

### 场景

朋友发了 `https://xiaoce.fun/guessword?shareId=123456789` 给你. 你想解这道.

### 步骤

```bash
# 方案 A: 用 HttpOracle 直接 (免 Playwright, 跟 v0.8.0 daily 一样)
# 这种方式简单, 不需要 Playwright, 适合任何场景
python3 << "PYEOF"
from sgs.wire.http import HttpOracle
from sgs.rank import load_corpus, rank, rank_by_predictor

# 1. 创建 oracle (anonymous, 跟 v0.8.0 daily 一样不需要登录)
oracle = HttpOracle(share_id="123456789")

# 2. 加载候选词
words, emb = load_corpus("data/cand_words.json", "data/cand_emb.npy")

# 3. 主动学习循环 (跟 sgs.daily_solve 同款)
obs = []
for round_idx in range(8):
    if len(obs) < 100:
        top = rank(obs, words, emb, top_k=30)
    else:
        top = rank_by_predictor(obs, words, emb, top_k=30)
    for w, _ in top:
        if any(o[0] == w for o in obs):
            continue
        r = oracle.probe(w)
        if r.score is None:
            print(f"err: shareId 不存在或接口限流, rate_limited={r.rate_limited}")
            continue
        obs.append((w, r.score))
        print(f"  probe {w:6s} = {r.score:.4f}")
        if r.correct:
            print(f"\\n✓ CORRECT: {w}")
            break
    if obs and obs[-1][1] > 0.99:
        break
PYEOF
```

### API 直接调用 (高级)

```python
# 单次探测 (验证 shareId 模式还活着)
from sgs.wire.http import HttpOracle
oracle = HttpOracle(share_id="123456789")
result = oracle.probe("广州")
print(f"广州 分数: {result.score}")  # 比如 0.71
print(f"正确?  {result.correct}")
```

完整自动求解脚本见上面的 bash 方案 A. `from sgs.wire.http import HttpOracle` 是公开 API, 想集成 sgs 的代码都可以用.

---

## 教程 D: 离线回放与 KRR 训练

> 10 分钟. 给 ML 工程师.

### 场景

你已经有 `case11_replay.ndjson` 这样的历史探测日志 (700+ 探次), 想离线训练 KRR 看效果.

### 步骤

```bash
# 1. 准备 replay 文件
ls data/case11_replay.ndjson  # 已存在, 701 个探次

# 2. 用 sgs.round1 跑离线 (不会发请求)
python -m sgs.round1 \
  --replay data/case11_replay.ndjson \
  --candidates data/cand_words.json \
  --embeddings data/cand_emb.npy \
  --batch-size 30 \
  --out data/case11_next.ndjson
```

### 启用 KRR predictor

```bash
# KRR 模式 (用 scikit-learn 的 KernelRidgeRegression, RBF 核, γ=0.1, α=0.1)
python -m sgs.round1 \
  --replay data/case11_replay.ndjson \
  --candidates data/cand_words.json \
  --embeddings data/cand_emb.npy \
  --predictor \
  --batch-size 30
```

期望: 9 探命中"继续" 0.9890. 跟 case-11 真实轨迹一致.

### 调 KRR 超参

```bash
python -m sgs.round1 \
  --replay data/case11_replay.ndjson \
  --candidates data/cand_words.json \
  --embeddings data/cand_emb.npy \
  --predictor \
  --gamma 0.5 --alpha 0.05 \
  --batch-size 30
```

γ 越大 → RBF 越局部 (适合稀疏 cluster). α 越大 → 越平滑 (适合小样本).

### Python 里直接调 KRR

```python
from sgs.krr import fit_predictor
from sgs.rank import load_corpus
import numpy as np

# 加载
words, emb = load_corpus("data/cand_words.json", "data/cand_emb.npy")

# 训练
import json
obs = [(json.loads(l)["word"], json.loads(l)["score"]) 
       for l in open("data/case11_replay.ndjson") 
       if json.loads(l).get("score") is not None]
X = np.array([emb[words.index(w)] for w, s in obs])
y = np.array([s for w, s in obs])

predictor = fit_predictor(X, y, gamma=0.1, alpha=0.1)

# 预测全部 corpus
preds = predictor(emb)
top10 = sorted(zip(words, preds), key=lambda x: -x[1])[:10]
for w, p in top10:
    print(f"{w}: predicted {p:.4f}")
```

---

## 教程 E: 自己造候选词库

> 20 分钟. 给数据工程师.

### 场景

默认的 `data/cand_words.json` 只有 2257 个常用 2 字词. 如果你解的题答案不在里面 (比如某个稀有专业词), 需要扩 corpus.

### 步骤

#### 1. 准备词表

```python
# my_words.py
my_words = [
    "青苔", "甘蔗", "砖瓦", "云霞",  # 文学词
    "原子", "量子", "光子", "电子",  # 物理词
    "布朗", "香山", "外滩", "陆家嘴",  # 上海地标
    # ... 加到 3000+ 个 2 字词
]
```

#### 2. 算 BGE 嵌入

**前置**: 安装 sentence-transformers

```bash
pip install sentence-transformers
# 国内网络下, 设镜像:
export HF_ENDPOINT=https://hf-mirror.com
```

#### 3. 嵌入脚本

```python
# embed_my_words.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-zh-v1.5")  # 768 维
words = my_words  # list[str]
print(f"loading {len(words)} words, encoding...")

# 关键: instruction 让 BGE 走对模式
# BGE 文档: "为这个句子生成表示以用于检索中文文档："
embeddings = model.encode(
    words,
    normalize_embeddings=True,  # 单位长度 — sgs 假设单位 norm
    show_progress_bar=True,
)
print(f"shape: {embeddings.shape}")

# 保存成跟 sgs 兼容的格式
np.save("data/cand_emb_custom.npy", embeddings.astype("float32"))
with open("data/cand_words_custom.json", "w") as f:
    json.dump(words, f, ensure_ascii=False)
print("✓ saved cand_emb_custom.npy + cand_words_custom.json")
```

```bash
python embed_my_words.py
# 输出: shape (3000, 768) ✓
```

#### 4. 用你的 corpus 跑 sgs

```bash
python -m sgs.daily_solve \
  --date 20260801 \
  --candidates data/cand_words_custom.json \
  --embeddings data/cand_emb_custom.npy \
  --batch-size 30 --rounds 6
```

### 检查 word 是否在 corpus

```python
# 验证 corpus + 嵌入匹配
import json
import numpy as np

words = json.load(open("data/cand_words_custom.json"))
emb = np.load("data/cand_emb_custom.npy")

assert len(words) == emb.shape[0], f"corpus 不匹配: {len(words)} words vs {emb.shape[0]} embeds"
assert emb.shape[1] == 768, f"BGE-base 是 768 维, 你的是 {emb.shape[1]}"
assert all(w != "" for w in words), "有空字符串"
assert all(len(w) == 2 for w in words), f"有非 2 字词: {[w for w in words if len(w) != 2][:5]}"
print("✓ corpus 健康")
```

---

## 常见问题

### Q1: daily_solve 跑了一会儿卡住, 还要不要等?

KRR 在稀疏 cluster 触发后**可能需要 100-300 探**才收敛. 给它 5-10 分钟. 如果 10 分钟还没结果, Ctrl-C 退出, 看 `out/...ndjson` 里的 top-3 候选, 自己用浏览器在 `xiaoce.fun/guessword` 试.

### Q2: rate_limit_exceed 错误

xiaoce 限流了. 默认 `rate=0.8 tokens/s` 是测出来不触限流的安全值. 如果你手动跑很多, 加 `--rate 0.5` 更保守.

### Q3: 答错了

几种可能:
- 答案不在 corpus 里 (走教程 E 加进去)
- 题目是 3 字 / 4 字 (sgs 假设 2 字) — 这种题需要扩展
- 题太冷门, BGE 没怎么见过 — 跑过概率低

### Q4: 怎么知道 cluster 类型是 dense 还是 sparse?

跑 `python -m sgs.daily_solve ...` 看前 30 探:
- 如果 top-1 分数 > 0.7 → **dense cluster** (城市/食物/地名/工具)
- 如果 top-1 < 0.5 → **sparse cluster** (抽象词/罕见词), KRR 会自动触发

### Q5: 我想看每探决策过程

```bash
# 跑 daily_solve 后, 读 ndjson 看完整轨迹
python3 << 'EOF'
import json
for line in open('/tmp/solve_20260801.ndjson'):
    d = json.loads(line)
    if d.get('correct'):
        print(f"✓ CORRECT: {d['word']} = {d['score']:.4f}")
    elif d.get('score') is not None:
        print(f"  probe {d['word']:6s} = {d['score']:.4f}")
    else:
        print(f"  err: {d.get('errCode', 'unknown')}")
EOF
```

### Q6: 我能把 ndjson 转成 Excel 看?

```bash
# 简单导出
python3 -c "
import json, csv
with open('/tmp/solve_20260801.ndjson') as fin, open('/tmp/solve.csv', 'w') as fout:
    csv.writer(fout).writerow(['word', 'score', 'correct', 'errCode'])
    for l in fin:
        d = json.loads(l)
        csv.writer(fout).writerow([d.get('word',''), d.get('score',''), d.get('correct',''), d.get('errCode','')])
"
# 然后用 Excel / Numbers 打开 /tmp/solve.csv
```

---

## 下一步

| 目标 | 文档 |
|---|---|
| 看项目整体故事 (26 小时从 0 到 8 个 tag) | `TECHNICAL.md` |
| 看 v0.8.0 daily 访客模式怎么发现的 | `TECHNICAL_DAILY_v0.8.0.md` |
| 想改 KRR 内部 | `sgs/krr.py` 源码 + 单元测试 `tests/test_krr.py` |
| 想加新功能 | `sgs/` 模块结构, `sgs/rank.py`, `sgs/probe.py` |
| 想部署生产 cron | 看 `cloud-vm-reverse-tunnel-deployment` skill (arkal.top 部署) |

仓库: [github.com/arckalsun/semantic-guess-solver](https://github.com/arckalsun/semantic-guess-solver)

有问题: [开 issue](https://github.com/arckalsun/semantic-guess-solver/issues)
