# semantic-guess-solver

> **26 小时连滚带爬**, 我做了一台能自动解 [xiaoce.fun GuessWord](https://xiaoce.fun)
> 每日猜词游戏的求解器. BGE 中文 embedding + 质心法 + 核岭回归 (KRR). 零依赖 (numpy only).
> 8 个 git tag (v0.1.0 → v0.8.0), 187 测试全绿, **完全免登录**就能跑 daily challenge.

[![tests](https://img.shields.io/badge/tests-187%20passed-brightgreen)]()
[![python](https://img.shields.io/badge/python-≥3.10-blue)]()
[![deps](https://img.shields.io/badge/runtime%20deps-numpy%20only-orange)]()
[![license](https://img.shields.io/badge/license-MIT-lightgrey)]()
[![repo](https://img.shields.io/badge/github-arckalsun%2Fsemantic--guess--solver-blue)]()

---

## 30 秒知道这是什么

```
猜词游戏: xiaoce.fun 每天出一道 2 字中文词, 你猜一个它给你一个分数 0-1, 1 = 答对
sgs 怎么猜: 拿 2257 个候选词的 BGE 嵌入, 用质心法 (centroid) 或 KRR 算出最值得猜的下一个词,
         然后发到 xiaoce API, 拿分, 重复, 直到 correct=true 或 探完
v0.8.0 关键: xiaoce 的 guessV1?date=YYYYMMDD 接口免登录 — daily challenge 完全可以 cron 自动化
```

> 不知道这是干嘛的? 看 [`TUTORIAL.md`](./TUTORIAL.md#5-分钟懂项目) 的 5 分钟懂项目.

---

## 🚀 5 分钟上手

```bash
# 1. 装
git clone https://github.com/arckalsun/semantic-guess-solver.git
cd semantic-guess-solver
pip install numpy

# 2. 跑今天的 daily (免登录, 免扫码, 免 cookie)
python -m sgs.daily_solve \
  --date $(date +%Y%m%d) \
  --candidates data/cand_words.json \
  --embeddings data/cand_emb.npy \
  --batch-size 30 --rounds 6 \
  --out /tmp/solve_$(date +%Y%m%d).ndjson

# 3. 等 1-7 分钟, 看结果
python3 -c "import json; print([l for l in open('/tmp/solve_$(date +%Y%m%d).ndjson') if json.loads(l).get('correct')][0])"
```

**输出示例**:
```
{'word': '分钟', 'score': 0.9890130573694068, 'correct': True}
```

完整教程 (4 套场景 + 6 个常见问题) 看 [TUTORIAL.md](./TUTORIAL.md).

---

## 🗂️ 项目结构

```
semantic-guess-solver/
├── README.md                # 你正在看
├── TUTORIAL.md              # 详细教程 (5 个场景 + 常见问题)
├── TECHNICAL.md             # 技术参考 (v0.7.0 KRR 数学 + 代码)
├── TECHNICAL_DAILY_v0.8.0.md  # v0.8.0 daily-mode 发现日志
├── sgs/                      # 核心库
│   ├── rank.py              # 排序 (centroid + KRR)
│   ├── krr.py               # 核岭回归预测器
│   ├── probe.py             # 单点探测 (URL 拼装 + 解析响应)
│   ├── oracle.py            # Oracle 协议 (fake/curl/playwright)
│   ├── ratelimit.py         # TokenBucket (0.8 tokens/s)
│   ├── replay.py            # NDJSON replay
│   ├── replay_diff.py       # replay 对比
│   ├── learn.py             # 主动学习循环
│   ├── flock.py             # 单实例锁 (防止 cron 并发)
│   ├── wire/                # HTTP + Playwright 实现
│   ├── round1.py            # CLI: rank-only 单批
│   ├── solve.py             # CLI: 完整 solver
│   └── daily_solve.py       # CLI: daily 访客模式 (v0.8.0)
├── tests/                    # 187 测试
│   ├── test_rank.py / test_krr.py / test_daily_solve.py
│   └── ... 17 个测试模块
├── data/                     # 资源
│   ├── cand_words.json      # 2257 个 2 字候选词
│   ├── cand_emb.npy         # BGE-base-zh 嵌入 (2257×768 float32, 单位 norm)
│   ├── cand_domain.json     # 候选词按域分类 (city/food/abstract/...)
│   ├── case11_replay.ndjson # case-11 完整 701 探次 replay (调试用)
│   └── daily/               # daily solve 的 NDJSON 日志
└── TECHNICAL.zh-CN.md         # TECHNICAL 中文版
```

---

## 📚 文档地图

| 文档 | 看什么 | 适合 |
|---|---|---|
| **`README.md`** (this) | 项目是什么 / 30 秒上手 / 5 分钟跑通 | 新用户第一次来 |
| **[`TUTORIAL.md`](./TUTORIAL.md)** | 4 个场景完整教程 (自动 daily / 解 shareId / 离线 KRR / 自造词库) + 6 个常见问题 | 想实际用起来的人 |
| **[`TECHNICAL.md`](./TECHNICAL.md)** | v0.7.0 KRR 数学 (5×5 网格 + 闭式解) + 触发规则 | ML 工程师 / 算法研究者 |
| **[`TECHNICAL_DAILY_v0.8.0.md`](./TECHNICAL_DAILY_v0.8.0.md)** | v0.8.0 daily 访客模式发现 + case-daily-20260715 | 想知道 daily 怎么免登录的 |

---

## 🤖 自动化 daily challenge (生产用)

如果你想**每天早 8 点自动跑 daily**, 配置 cron:

```bash
# 1. 准备 run.sh
cat > ~/sgs-daily/run.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d)
python -m sgs.daily_solve \
  --date ${DATE} \
  --candidates ~/sgs-daily/cand_words.json \
  --embeddings ~/sgs-daily/cand_emb.npy \
  --batch-size 30 --rounds 8 \
  --out ~/sgs-daily/solve_${DATE}.ndjson
EOF
chmod +x ~/sgs-daily/run.sh

# 2. 加 cron
crontab -e
# 加: 0 8 * * * ~/sgs-daily/run.sh >> ~/sgs-daily/cron.log 2>&1
```

完整步骤 + 验证 / 监控 / 异常处理看 [TUTORIAL.md 教程 B](./TUTORIAL.md#教程-b-自动化-daily-challenge).

---

## 🔧 已实现的 8 个版本里程碑

| Tag | 日期 | 关键功能 |
|---|---|---|
| v0.1.0 | 7-14 09:59 | Round 1 numpy ranker (质心法雏形) |
| v0.2.0 | 7-14 10:12 | Oracle 协议 + TokenBucket + 限速 |
| v0.3.0 | 7-14 13:02 | Playwright 兜底 + 单实例锁 + 主动学习 |
| v0.4.0 | 7-14 18:26 | 完整 solver CLI |
| v0.5.0 | 7-14 18:35 | replay_diff 离线回归 |
| v0.6.0 | 7-14 18:40 | opt-in live wire integration |
| **v0.7.0** | 7-15 06:23 | **KRR 9 探破 case-11 0.82 plateau** |
| **v0.8.0** | 7-15 12:38 | **daily 访客模式 (免登录)** |

---

## ❓ 常见问题 (完整版见 [TUTORIAL.md](./TUTORIAL.md#常见问题))

**Q: 我没有任何基础能跑起来吗?**
A: 能. 只要有 Python 3.10+, 跑上面 3 行 `git clone` + `pip install` + 跑命令就行.

**Q: 不登录能用吗?**
A: 能. v0.8.0 之后 daily 模式**完全免登录**. shareId 模式也免登录 (你拿到 shareId 直接跑).

**Q: 答错了怎么办?**
A: 几种可能: (1) 答案不在 corpus (走 [教程 E](./TUTORIAL.md#教程-e-自己造候选词库) 加词); (2) 题目不是 2 字词; (3) 题太冷 BGE 不认识. 前 30 探分数 < 0.5 就是第 3 种, 加 corpus 也未必管用.

**Q: 能解别人分享给我的题吗?**
A: 能. 教程 C: 拿 `shareId` 跑 `sgs.solve`. 看 [教程 C](./TUTORIAL.md#教程-c-解-shareid-分享题).

**Q: 怎么知道 cluster 类型 (dense vs sparse)?**
A: 跑前 30 探: top-1 分数 > 0.7 = dense (城市/食物/地名), < 0.5 = sparse (抽象词), KRR 自动触发.

---

## 📋 验证清单 (跑通后自检)

- [ ] `python -m pytest -q` → 187 passed
- [ ] `python -c "import json; print(len(json.load(open('data/cand_words.json'))))"` → 2257
- [ ] `python -c "import numpy as np; print(np.load('data/cand_emb.npy').shape)"` → (2257, 768)
- [ ] `python -m sgs.daily_solve --help` 列出所有 flag
- [ ] 跑 `python -m sgs.daily_solve --date 20250720 ...` (历史日期) 验证 corpus 健康
- [ ] 跑当天 daily, 在 5-10 分钟内出 correct=true

---

## 🤝 贡献 / 反馈

- 仓库: [github.com/arckalsun/semantic-guess-solver](https://github.com/arckalsun/semantic-guess-solver)
- 提 issue: 描述场景 + 命令 + 输出 (不要只贴"不工作了")
- 加 PR: 跑 `pytest -q` 187 passed + 简单复现 case 才有意义

## 📜 License

MIT — 改、卖、商用都行, 留个 author 即可.

---

**项目地址**: [github.com/arckalsun/semantic-guess-solver](https://github.com/arckalsun/semantic-guess-solver) ·
**最新 tag**: v0.8.0 ·
**License**: MIT
