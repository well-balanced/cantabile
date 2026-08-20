# Human-Human Velocity MAE Anchor

## 이게 왜 필요한가

논문은 velocity MAE를 26.4 → 7.0 (3.8배 개선, abstract 기준)으로 보고한다.
근데 이 숫자엔 **절대적인 기준점(anchor)이 없다.** 7.0이 "거의 완벽한"
수준인지 "아직 갈 길이 먼" 수준인지, 비율만으로는 판단할 수 없다.

> **별개로 발견한 것**: abstract는 baseline MAE를 26.4라고 쓰는데, 부록
> Component Ablation Table(`main.tex:452`)은 같은 조건을 26.2로 보고한다.
> 둘 다 논문에 실제로 있는 값이고 이 문서가 옮기며 생긴 오차가 아니다 —
> 논문 자체의 abstract-vs-표 불일치이니 다음 리비전에서 정리할 것. 아래
> 비교표들은 부록 표 값(26.2)을 기준으로 한다.

Reviewer vmh3의 지적이 정확히 이 지점이다:

> "어느 수준(threshold)부터 그것이 유의미한지가 불분명하게 남아 있다. 메트릭
> 자체도 논문의 기여 중 하나이므로..."

이 디렉토리는 그 기준점을 만드는 실험 중 가장 싸고 빠른 것 — **인간 연주자
사이의 자연스러운 velocity 편차**를 MAESTRO 데이터셋에서 직접 계산한다.
사람도, 학습도, GPU도 필요 없고 이미 존재하는 MIDI를 열어서 숫자를 빼는
수준의 작업이다.

## 원리

같은 곡을 서로 다른 연주자가 치면, 악보의 강약 지시(forte/piano 등)는
같아도 실제 velocity 값은 사람마다 다르다. 이 자연스러운 차이의 크기를
알면:

- 로봇의 오차가 그보다 **크면** → 아직 사람이 듣기에 명백히 이상한 수준
- 로봇의 오차가 그보다 **작으면** → 두 사람의 해석 차이보다도 로봇이 레퍼런스에
  가깝다는 뜻 → 강한 근거

## 데이터: MAESTRO v3.0.0

[MAESTRO](https://magenta.tensorflow.org/datasets/maestro) (Magenta,
Google) — International Piano-e-Competition 실황을 Disklavier로 녹음한
실제 연주 MIDI 데이터셋. 1276개 연주, `maestro-v3.0.0.csv`에 곡 제목
(`canonical_title`), 작곡가, 연도, 파일명이 있다.

**주의**: MAESTRO는 연주자 이름/ID를 공개하지 않는다 (프라이버시 목적).
그래서 "다른 사람이 쳤다"를 직접 확인할 방법이 없고, 대신 **같은
`canonical_title`을 가진, 연도가 다른 녹음**을 서로 다른 연주로 간주한다
(같은 콩쿠르가 매년 다른 참가자로 열리므로 합리적인 근사지만, 100% 보장은
아니다 — 아래 한계 참고).

## 방법론 (`compute_human_mae.py`)

1. **곡 그룹핑**: `canonical_title`이 같은 연주가 2개 이상인 곡만 사용
   (1276개 중 204개 제목, 곡당 최대 18개 연주까지 존재).
2. **쌍 선택**: 곡당 조합 폭발을 막기 위해 title당 최대 `cap-per-title`(기본
   6)쌍만 샘플링하고, 연도가 다른 쌍을 우선 배치한다. 전체 `max-pairs`
   (기본 350)로 캡.
3. **음 추출**: `pretty_midi`로 각 연주의 (onset, pitch, velocity)를
   onset 시각 순으로 정렬.
4. **매칭 — 시간이 아니라 음높이 순서로**: 두 연주는 템포가 달라서 절대
   시각으로 비교할 수 없다. 대신 같은 악보를 치는 거라 **음높이 시퀀스가
   거의 동일**하다는 사실을 이용해 `difflib.SequenceMatcher`로 두 pitch
   시퀀스의 최장 공통 부분열(LCS)을 찾는다. 이게 음-대-음 대응 관계가
   된다 — 시간 정보 없이도 정렬 가능.
5. **필터링**: 매칭된 음이 10개 미만이거나, `match_frac`(매칭 수 /
   min(두 연주 음 개수))이 0.3 미만인 쌍은 버린다 — 이런 경우는 대개 같은
   제목이라도 다른 악장/발췌인 경우다.
6. **MAE 계산**: 매칭된 음 쌍마다 `|velocity_a - velocity_b|`, raw MIDI
   velocity 스케일(0-127) — `env/robopianist/wrappers/evaluation.py`의
   `velocity_mae`와 동일 단위. (`inspect_song_velocities.py`로 확인한
   3곡의 평균 velocity가 논문 `VBAR`값과 거의 일치하는 것으로 단위 일관성
   재확인함, 아래 참고.)

## 실행 방법

```bash
.venv/bin/python rl/human_mae/compute_human_mae.py
```

첫 실행 시 MAESTRO MIDI-only 아카이브(~58MB, public, CC BY-NC-SA)를
`rl/tmp/maestro/`(gitignored)에 자동 다운로드한다. 이후 실행은 캐시를
재사용한다. 결과는 `results/human_mae_pairs.json`(쌍별 상세)과
`results/human_mae_summary.json`(집계)에 저장된다.

관련 스크립트 `inspect_song_velocities.py`는 논문이 실제로 쓰는 3곡
(Twinkle, Nocturne, Clair de Lune)의 GT velocity가 상수 박제가 아니라
진짜 연주 데이터인지 확인한다 (vmh3의 "velocity target이 얼마나 자주
있나" 질문에 대한 직접 답).

## 결과

`results/human_mae_summary.json` 참고. 요약(seed=0, cap-per-title=6,
max-pairs=350 기준):

| | 값 |
|---|---|
| 사용된 쌍 (정렬 품질 통과) | 299 / 350 |
| 곡 제목 수 (사용된 쌍 기준) | 127 |
| 매칭된 음 쌍 (전체 pooled) | ~1,047,000 |
| **쌍별 MAE — mean** | **10.78** |
| 쌍별 MAE — median | 10.64 |
| 쌍별 MAE — std | 1.72 |
| 쌍별 MAE — p10 / p90 | 8.73 / 12.95 |
| Pooled MAE (전체 음 쌍 기준) | 10.98 |

분포가 좁게 모여있다(std 1.7) — 우연이 아니라 안정적인 신호로 볼 수 있다.

### 3곡 GT velocity 소스 확인 (`song_velocity_stats.json`)

| 곡 | 소스 | 음 개수 | unique velocity | mean | std |
|---|---|---|---|---|---|
| Twinkle (Rousseau) | 실제 유튜버 연주, 허가받아 사용 | 34 | 21 | 40.3 | 8.2 |
| Nocturne (Rousseau) | 실제 유튜버 연주, 허가받아 사용 | 117 | 32 | 39.6 | 14.1 |
| Clair de Lune (PIG) | PIG 데이터셋(실연주 기반) | 79 | **7** | 44.2 | **4.3** |

셋 다 상수 velocity가 아니라 진짜 연주 데이터. 다만 Clair de Lune은
unique velocity 값이 7개뿐이고 std도 확연히 작다 — PIG 쪽 데이터가
상대적으로 덜 세밀하거나(quantization), 곡 자체가 발췌 구간 내내 여리게
(pp) 연주되는 부분이라 그럴 수 있다. 논문에 이 이질성을 숨기지 말고
명시하는 게 낫다. 논문 `VBAR = {twinkle: 39.0, clair: 43.8, nocturne:
43.0}`과 비교해도 대체로 일치해 — raw MIDI velocity 스케일이 맞다는
교차검증이 된다 (Nocturne 쪽만 약간 차이나는데, 트리밍 구간이나 v̄ 계산에
포함된 음 집합이 조금 다를 수 있음 — 필요하면 추가 확인).

## 왜 matched-only MAE를 그대로 비교하면 안 되는가

논문의 Vel MAE(26.2, 7.0 등)는 **matched onset에서만** 계산된다(§4.1: "velocity
MAE is computed only over matched onsets"). Baseline은 onset F1이 0.687밖에
안 돼서 — 대략 GT onset의 22~32%가 매칭에서 빠진다. 이게 무작위로 빠지는 게
아니라 **체계적으로 어려운 음(빠른 패시지, 밀집 화음, 장식음)이 더 많이
빠질 가능성이 높다.** 그러면 matched-only MAE는 "쉬운 음만 골라 잰" 낙관적인
숫자가 되고, 이걸 인간-인간 앵커(사실상 전체 음 커버)와 그냥 나란히 놓는 건
표본 구성이 다른 두 숫자를 비교하는 것이다. (이게 바로 논문 §4.1이 Dynamics
Score를 만든 이유이기도 하다 — 근데 D는 MAE를 커버리지로 *가중*만 할 뿐,
MAE 자체의 편향은 고치지 않는다.)

## Recall-Weighted MAE (`recall_weighted_mae.py`)

그래서 matched-only MAE를 **전체 GT onset 기준**으로 환산한다. 놓친 온셋마다
페널티를 부과하는데, 이 페널티로 논문이 이미 D의 정의에서 "완전히 놓쳤을
때"의 기준으로 쓰는 `v̄`(평균 GT velocity)를 그대로 재사용한다:

```
MAE_rw = R_onset * MAE_matched + (1 - R_onset) * v̄
```

`R_onset = N_hit / N_gt` (onset recall, wandb의 `eval/onset_hit_rate`).
이렇게 하면 인간-인간 앵커와 **같은 단위, 같은 커버리지 기준**으로 비교
가능해진다.

**데이터 소스는 두 단계**: ① `rl/analysis/fetch_wandb.py`(METRICS에
`eval/onset_hit_rate` 이미 추가해둠)로 받은 실제 per-run `onset_hit_rate`가
있으면 그걸 정확히 쓴다. ② wandb 접근이 없으면(지금 이 환경처럼)
논문 부록 Component Ablation Table의 공개된 곡-평균 수치로 대체하되, 그
표엔 recall이 따로 없어서 **onset F1을 recall의 근사치로 사용한다**
(precision≈recall일 때만 정확 — 출력에 `approx: true`로 항상 명시됨).

### 실행

```bash
.venv/bin/python rl/human_mae/recall_weighted_mae.py
```

### 결과 (`results/recall_weighted_comparison.json`, 현재는 approx 모드)

| Condition | MAE (matched-only) | MAE (recall-weighted) |
|---|---|---|
| Baseline | 26.2 | **31.1** |
| Vel-Aware | 12.9 | 19.3 |
| Base+Res | 10.6 | 19.7 |
| **Vel-Aware+Res (ours)** | 7.0 | **15.0** |
| 인간-인간 앵커 | — | **10.8** |

### 결론이 바뀐다 — 근데 더 방어 가능한 쪽으로

naive 비교(matched-only 7.0 < 앵커 10.8)로는 "로봇이 인간보다 정확하다"고
읽히는데, **recall-weighted로 보면 그 주장은 성립하지 않는다** (15.0 >
10.8). 대신 정직하고 여전히 강한 주장은:

> Baseline은 coverage를 반영해도 인간 자연 변동폭보다 확실히 나쁘다(31.1 vs
> 10.8). METHOD는 그 갭을 대부분 좁혔지만(31.1 → 15.0), 아직 인간 수준에는
> 못 미친다(15.0 vs 10.8).

naive 비교로 "인간보다 낫다"고 냈으면 리뷰어가 정확히 이 지점(matched-only
selection bias)으로 반박했을 것이다. 이 보정판이 그 공격을 막으면서도
baseline 대비 개선폭(3배 이상 갭 축소)은 그대로 보여준다.

**중요**: 지금 표는 `approx: true` — onset_f1을 recall 근사로 쓴 것이다.
`rl/analysis/fetch_wandb.py`를 wandb 인증 되는 곳에서 먼저 돌리고(이미
`eval/onset_hit_rate`를 받아오도록 고쳐둠), 그 다음 이 스크립트를 다시
돌리면 실제 `onset_hit_rate`로 정확한 숫자가 나온다. 논문에 넣기 전에
반드시 이 정확한 버전으로 교체할 것.

### 반드시 명시할 비대칭성 (matched-only 숫자를 어디서든 인용한다면)

recall-weighted 버전은 "인간보다 낫다"고 주장하지 않으니 이 문제를 자동으로
피해가지만, matched-only 숫자(7.0 등)를 논문 어디서든 인용할 경우엔 여전히
밝혀야 한다: 두 인간 연주자는 **각자 독립적으로 자유 해석**한 것이고, 로봇은
`velocity_goal` look-ahead observation으로 **정답을 미리 보고 그걸 추적**한다.
그러므로 matched-only MAE가 앵커보다 낮게 나오는 것도 "로봇이 인간보다
표현력이 뛰어나다"가 아니라 "주어진 레퍼런스를 향한 실행 오차가 작다"는 좁은
주장으로만 읽혀야 한다.

## 한계 (정직하게 밝혀야 할 것)

1. **연주자 신원 불명**: MAESTRO가 performer ID를 안 줘서 "다른 연도 =
   다른 사람"은 근사일 뿐, 보장되지 않는다. 같은 사람이 여러 해에 출전했을
   가능성은 배제 못 함.
2. **정렬은 휴리스틱**: `difflib` 기반 pitch-order 매칭은 쌍당 평균
   **64%**의 음만 매칭한다(화음 내부는 순서가 원래 모호하고, 반복구·장식음
   차이 때문에 나머지가 빠짐). 100만+ 쌍을 모아 통계적으로는 안정적이지만,
   camera-ready 단계에서는 PIG 데이터셋이 쓰는 것 같은 제대로 된 symbolic
   alignment(DTW) 도구로 재검증하면 더 방어력이 세진다.
3. **레퍼토리 이질성**: MAESTRO는 콩쿠르용 대곡 위주라 논문의 3곡(비교적
   짧고 단순)과 스타일이 다를 수 있다. 필요하면 `VBAR`로 정규화한 비교나,
   레퍼토리가 더 가까운 서브셋으로 제한한 버전을 추가로 돌릴 것.
4. **`match_frac` 낮은 쌍도 포함**: 0.3 이상이면 통과시켰는데, 이 임계값
   자체는 다소 임의적이다. sensitivity check로 0.5, 0.7 기준에서 결과가
   크게 안 바뀌는지 확인하면 더 견고해진다 (아직 안 함).
5. **놓친 온셋 페널티로 `v̄`를 쓰는 게 임의적 선택**: "완전히 놓치면 v̄만큼
   틀린 걸로 친다"는 논문의 D 정의를 그대로 재사용한 것이지, 유일하게
   타당한 값은 아니다. 더 가혹한 페널티(예: 관측된 velocity의 최댓값)를
   쓰면 recall-weighted MAE는 더 커진다 — 즉 지금 15.0/31.1은 이
   가정 하의 한 버전이고, 페널티를 바꿔가며 결과가 얼마나 민감한지
   보여주면 더 견고해진다.
6. **fallback 모드는 근사치**: `eval/onset_f1`을 recall 대용으로 쓴 것 —
   precision과 recall이 실제로 다르면 오차가 생긴다. 논문에 넣기 전
   `rl/analysis/fetch_wandb.py`로 받은 실제 `onset_hit_rate`로 반드시
   재계산할 것.

## 파일 구성

```
rl/human_mae/
  README.md                       # 이 문서
  compute_human_mae.py            # 인간-인간 MAE 계산 (메인)
  inspect_song_velocities.py      # 3곡 GT velocity 소스/분포 확인
  recall_weighted_mae.py           # matched-only MAE -> 전체 GT onset 기준 환산 + 앵커 비교
  results/
    human_mae_pairs.json               # 쌍별 상세 결과 (title, 연도, matched 수, MAE)
    human_mae_summary.json             # 집계 통계
    song_velocity_stats.json           # 3곡 velocity 분포
    recall_weighted_comparison.json    # recall-weighted MAE 비교 결과 (현재 approx 모드)
```

`rl/tmp/maestro/`(MAESTRO 캐시)와 `rl/tmp/wandb_data/`(fetch_wandb.py
출력)는 둘 다 gitignored — 리포에 안 들어감, 스크립트 재실행 시 자동
재생성/재사용됨.

## 실행 순서 요약

```bash
# 1. 인간-인간 앵커 (MAESTRO 다운로드 자동, ~58MB, 첫 실행만 느림)
.venv/bin/python rl/human_mae/compute_human_mae.py

# 2. 3곡 GT velocity 소스 확인
.venv/bin/python rl/human_mae/inspect_song_velocities.py

# 3. (wandb 인증 되는 곳에서) 실제 조건별 onset_hit_rate/velocity_mae 수집
.venv/bin/python rl/analysis/fetch_wandb.py

# 4. recall-weighted MAE 계산 + 앵커와 비교 (3번 실행했으면 exact, 아니면 approx)
.venv/bin/python rl/human_mae/recall_weighted_mae.py
```
