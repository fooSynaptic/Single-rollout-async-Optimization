# MATH-hard experiment report

> Date: 2026-07-25 · curves: [docs/figures/math-hard-acc.png](figures/math-hard-acc.png), [docs/figures/math-hard-reward.png](figures/math-hard-reward.png)  
> Metric: online `acc = correct / (correct + incorrect)`  
> Model: `Qwen3-4B-Instruct-2507` · data: MATH → GSM8K-style schema (`scripts/prepare_math.py`) · cap **1000** steps

## 1. What this rep wanted to test

| Goal | Meaning |
|------|---------|
| Hard task separates methods | Opening reward / acc should be far below GSM8K saturation (~92–95%) |
| Stability | DIS / SAO should resist collapse; bare GRPO and bare running-mean should not |
| Fair same-budget comparison | Report **G=1 vs G=8** separately; under G=1 compare SAO vs single-traj+DIS vs running-mean |
| Out of scope | Paper table absolute scores · held-out MATH-500 / AIME (not run here) |

## 2. What tasks ran

| Setting | Factors | Outcome | End-window online acc |
|---------|---------|---------|------------------------|
| **SAO** | G=1 · critic · DIS | **1000/1000** | **~49%** (951–1000) |
| **GRPO** | G=8 · group · no DIS | **early-stop @268** | **~15%** (251–268) |
| **GRPO+DIS** | G=8 · group · DIS | **1000/1000** | **~57%** (951–1000) |
| **running-mean** | G=1 · running window · no critic · no DIS | **1000/1000** | **~10%** (951–1000) |
| **single-traj + DIS** | G=1 · batch norm · DIS · no critic (config name `grpo_dis_g1`; **not GRPO**) | **1000/1000** | **~43%** (951–1000) |

Order: SAO → GRPO → GRPO+DIS → running-mean → single-traj+DIS (serial).

## 3. Windowed online accuracy (%)

### 3.1 Main comparison (includes G=8)

| Window (step) | SAO | GRPO | GRPO+DIS | running-mean |
|---------------|-----|------|----------|--------------|
| 1–50 | 39.9 | 38.9 | 40.2 | 39.7 |
| 51–100 | 40.7 | 35.8 | 43.2 | 38.4 |
| 200–268 | 41.0 | **21.3** | 46.7 | 30.1 |
| 251–268 | 43.1 | **15.4** | 47.1 | 30.9 |
| 401–500 | 42.9 | — | 51.3 | 24.5 |
| 501–600 | 43.5 | — | 52.3 | 21.0 |
| 701–800 | 45.6 | — | 54.9 | 12.6 |
| 851–900 | 47.2 | — | 55.8 | 15.9 |
| 901–950 | 47.8 | — | 56.1 | **5.4** |
| **951–1000** | **49.3** | — | **57.2** | **10.5** |

### 3.2 Same G=1 budget (single-traj + DIS)

| Window (step) | SAO | single-traj + DIS | running-mean |
|---------------|-----|-------------------|--------------|
| 1–50 | 39.9 | 39.8 | 39.7 |
| 51–100 | 40.7 | 40.1 | 38.4 |
| 401–500 | 42.9 | 40.9 | 24.5 |
| 701–800 | 45.6 | 42.6 | 12.6 |
| 851–900 | 47.2 | 43.4 | 15.9 |
| 901–950 | 47.8 | 42.1 | **5.4** |
| **951–1000** | **49.3** | **43.5** | **10.5** |

Opening accuracy for all settings is **~39–40%**: the hard task surface holds.

## 4. Conclusions

### 4.1 Stability

1. **Bare GRPO collapses early**: ~39% → ~15%, early-stop @268.  
2. **DIS is a strong stabilizer**: G=8 + DIS only → steady rise to ~57%.  
3. **SAO (G1 + critic + DIS) does not collapse**: 40% → 49%.  
4. **running-mean slowly collapses**: ~10% at the end.  
5. **single-traj + DIS also holds**: ~43% end — DIS helps at G=1, but is weaker than SAO with a critic.

Rough stability order: **SAO ≈ GRPO+DIS (G8) ≈ single-traj+DIS ≫ running-mean > GRPO (dies early)**.

### 4.2 Scores (keep G=1 / G=8 separate)

- **Highest absolute score is G=8**: GRPO+DIS ~57% (8 trajectories per step).  
- **Same G=1 budget**: **SAO ~49% > single-traj+DIS ~43% ≫ running-mean ~10%**.  
- Critic+GAE does **not** beat “G=8 + DIS” on absolute online score; G=8 still enjoys a sampling budget advantage.

### 4.3 Claims

| Claim | Evidence here |
|-------|----------------|
| SAO’s main value is single-rollout + stability | ✓ G=1 rises steadily; bare G=8 collapses |
| DIS mainly prevents collapse / keeps usable gradients | ✓ holds on both G=8 and G=1 lines |
| Critic beats “DIS only” under the same budget | ✓ 49% > 43% |
| running-mean can replace group / critic | ✗ slow collapse |
| single-traj + DIS may be called GRPO | ✗ G=1 has no group-relative advantage |

## 5. One-line summary

On **MATH-hard · 4B Instruct · 1k steps**:

- **DIS + (G1+critic or G8) can run hard MATH stably; bare GRPO / bare running-mean cannot.**  
- **Same G=1: SAO > single-traj+DIS ≫ running-mean.**  
- **Best absolute online score remains G=8 GRPO+DIS** — do not treat that as “strictly better algorithm” than G=1 methods.

## 6. Future steps

1. **Held-out eval**: MATH-500 / AIME.  
2. **Generation budget**: raise `max_new_tokens` / `max_model_len` toward common math-RL settings.  
3. **Larger models**: only after eval + budget alignment; do not scale to 30B from online 49% alone.
