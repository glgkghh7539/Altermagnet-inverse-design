# NiS Formation Energy — Ni–S Convex Hull

NiAs형 NiS(α-NiS, 알터마그넷 후보)가 열역학적으로 어디에 놓이는지를 정량화한 작업.
리뷰어 대응이 목적이며, **Materials Project와 직접 비교 가능한 형태**로 값을 낸다.

- 서버: TGM (`ssh MASTER`), 작업 경로 `~/NiS_hull/`
- 계산: VASP 6.4.2 · 9개 상 (본 hull) + 40개 런 (Hubbard U 스캔)
- 상태: **완료**

---

## 1. 핵심 결과

**NiAs형 NiS는 0 K hull 위 {nias_hull:.1f} meV/atom, Ef = {nias_ef:.4f} eV/atom** (MP2020 보정 기준).

저온 안정상인 millerite는 hull 위 {mil_hull:.1f} meV/atom이므로, 두 다형체 간격은
**{poly_gap:.0f} meV/atom** (millerite가 낮음) — 실험과 일치한다.

![convex hull](fig1_convex_hull.png)

---

## 2. 계산 조건

모든 상에 **동일 조건**을 적용했다. 이것이 formation energy 비교를 성립시키는 전제다.

| 항목 | 값 |
|---|---|
| 범함수 | PBE, **Hubbard U 없음** (MP와 비교 가능하게 유지) |
| POTCAR | PAW 54, `Ni_pv` / `S` |
| ENCUT | 520 eV, PREC=Accurate, LASPH, LREAL=.FALSE. |
| k-점 | **KSPACING**: relax 0.25 / static 0.15, KGAMMA=.TRUE. |
| relax | ISIF=3, IBRION=2, NSW=120, EDIFFG=−0.01, ISMEAR=1/σ=0.2, **2-pass** |
| static | NSW=0, ISMEAR=−5 (tetrahedron), EDIFF=1e-6, LORBIT=11 |
| 스핀 | ISPIN=2, FM 초기화 (MAGMOM Ni 5.0 / S 0.6) |

k-점을 KSPACING으로 잡은 이유: 상마다 셀 크기가 다르므로(1원자 Ni부터 34원자 Ni₉S₈까지)
**동일한 k-점 밀도**를 보장해야 한다. 상별로 격자를 임의 지정하면 비교가 성립하지 않는다.
실제 유도된 격자는 Ni의 21×21×21(static)부터 α-S의 5×5×5까지 걸쳐 있다.

2-pass relax는 ISIF=3의 평면파 basis가 초기 셀에 고정되는 문제(Pulay stress) 때문이다.
pass1 완화 후 CONTCAR에서 재시작하면 현재 부피 기준으로 basis가 재생성된다.

---

## 3. 전체 결과

{hull_table}

★ = hull 위(안정). Ef는 eV/atom, E_hull은 meV/atom.
`Ef_raw`는 순수 PBE 값, `Ef (MP2020)`은 MP2020 황화물 보정을 적용한 값.

**안정상**: Ni, Ni₃S₂, Ni₉S₈, Ni₃S₄, NiS₂(P2₁/c), S — MP와 동일한 집합.

준안정상 분해 경로:

```
NiS millerite    15.0 meV/atom  ->  0.708 Ni9S8 + 0.292 Ni3S4
NiS2 pyrite      15.6 meV/atom  ->  NiS2 (P2_1/c로 다형체 전이)
NiS NiAs형      106.8 meV/atom  ->  0.708 Ni9S8 + 0.292 Ni3S4
                                    (= 1 NiS -> 1/12 Ni9S8 + 1/12 Ni3S4)
```

---

## 4. MP2020 보정 — 이 작업의 핵심 발견

순수 PBE로 얻은 Ef가 MP 값보다 **정확히 S 원자당 0.503 eV씩 높았다.**
이 값은 Materials Project의 MP2020 황화물 음이온 보정과 일치한다 (pymatgen에서 직접 확인).

보정을 적용하면 {n_dev}개 상에 대해 **평균 {mad:.1f} meV/atom** (범위 {dev_lo:+.1f} ~ {dev_hi:+.1f})으로
MP를 재현한다. E_hull도 맞는다 — millerite 15.0 vs MP 14.2, pyrite 15.6 vs 15.2,
NiAs형 106.8 vs mp-594의 104.7.

![MP validation](fig2_mp_validation.png)

> **논문에 쓸 값은 `Ef (MP2020)` 열이다.** raw PBE 값을 그대로 MP와 비교하면
> S 조성에 비례하는 계통 오차가 그대로 남는다.

---

## 5. E_hull은 기준물질에 무관하다

Ef는 기준물질(μ_Ni, μ_S) 선택에 따라 움직이지만, **E_above_hull은 움직이지 않는다.**

μ_S에 오차 δ가 있으면 모든 Ef가 −x_S·δ만큼 이동한다. 이는 조성에 대한 1차(affine) 함수이고,
convex hull 구성은 affine 변환에 불변이므로 hull 정점 집합도 각 상의 hull 위 거리도 그대로다.

수치 확인 (`scripts/mu_test.py`) — μ_S를 ±0.3 eV 강제로 흔들었을 때:

```
  d(mu_S)   NiAs형 Ef      E_hull    |  안정상 집합
   -0.30    -0.1159      106.8 meV  |  동일
    0.00    -0.2659      106.8 meV  |  동일
   +0.30    -0.4159      106.8 meV  |  동일
```

**따라서 α-S를 vdW 없이 PBE로 계산한 것은 결론에 영향이 없다.**
α-S는 S₈ 분자결정이라 vdW 없는 PBE에서 셀이 부풀지만(실험 대비 +38%),
그 오차는 μ_S에만 들어가고 E_hull에서는 상쇄된다. 게다가 MP도 mp-77을 동일하게
순수 PBE로 계산하므로, vdW를 넣으면 오히려 MP와의 비교가 깨진다.

단, **원소 S가 평형에 직접 등장하는 계산**(S 분압, NiS₂ → NiS + S 분해 등)에서는
μ_S가 상쇄되지 않으므로 vdW가 중요해진다. 이번 경우 NiAs형의 분해 경로에는
원소 S도 Ni 금속도 등장하지 않는다.

---

## 6. Hubbard U 검토 — 열역학에는 쓰면 안 된다

포논 계산이 U=7이므로 hull도 U를 넣어야 하는지 검토했다. **결론: 넣으면 안 된다.**

E_hull이 Ni₉S₈–Ni₃S₄ tie-line 거리이므로 원소 기준물질이 상쇄된다는 성질을 이용해,
**Ni 황화물 7개에만** U를 걸어 스캔했다 (Ni 금속에 U를 거는 비물리적 상황을 피할 수 있다).
U=0, 2, 4, 6, 7 × {{7개 황화물 FM + NiAs형 AFM}} = 40런.

자체 검증: U=0에서 황화물만으로 얻은 값이 106.7 / 106.8 meV/atom으로,
9상 전체 hull의 106.8과 일치한다.

{u_table}

단위 meV/atom. **음수는 NiAs형이 분해 산물보다 안정하다는 뜻이며 물리적으로 틀렸다** —
α-NiS는 고온상이라 0 K에서 안정할 수 없다.

### 다형체 서열 — 결정적 근거

실험적으로 millerite가 ~379 °C 아래에서 안정상이다.

{order_table}

**U=0만 실험을 맞춘다.** U≥2에서는 고온상이 저온상보다 안정하다고 예측한다.

### 격자 부피

실험 α-NiS는 약 13.7 Å³/atom (a≈3.44, c≈5.35 Å).
U=0이 13.28(−3%)로 가장 가깝고, U가 커질수록 모멘트 발현과 함께 부풀어
U=7에서 15.26(+11%)까지 벌어진다.

![U dependence](fig3_U_dependence.png)

### 종합

| 기준 | U=0 | U≥2 |
|---|:---:|:---:|
| MP와의 일치 ({mad:.1f} meV/atom) | ○ | 비교 불가 |
| millerite < NiAs형 (실험) | ○ | ✗ 역전 |
| 격자 부피 | −3% | +2~11% |

**열역학은 U=0, 자성·포논은 U=7.** 두 결과는 병렬 제시하되 자유에너지를 섞지 말 것.

---

## 7. 자성 상태

hull의 모든 황화물은 FM 초기화 후 **무자성으로 수렴**했고, fcc Ni만 0.63 μB를 유지했다.
U 없는 PBE에서 예상되는 거동이며 MP의 처리와 동일하다.

NiAs형 NiS를 AFM(Ni z=0 up / z=½ down, 포논 계산과 동일 배열)으로 별도 계산한 결과:

| 상태 | U | E₀ (eV/4원자) | V (Å³) | m(Ni) μB |
|---|---:|---:|---:|---:|
| NM | 0 | −20.31556799 | 53.10 | 0.000 |
| AFM | 0 | −20.31544426 | 53.10 | **0.058** |
| AFM | 7 | −13.60008236 | 61.15 | **1.626** |
| FM | 7 | −13.39569832 | 62.43 | 1.728 |
| NM | 7 | −10.64006705 | 52.34 | 0.000 |

**U=0에서 AFM은 사실상 존재하지 않는다** — 모멘트 0.058 μB, 에너지는 NM보다
0.031 meV/atom 오히려 높다(잡음 수준의 축퇴). 따라서 {nias_hull:.1f} meV/atom은
상한이 아니라 확정값이다.

U=7에서는 견고한 AFM(1.63 μB)이 나오며, FM보다 51 meV/atom, NM보다 740 meV/atom 낮다.
셀 팽창은 U가 아니라 **모멘트 형성**이 일으킨다 — U=7에서도 NM은 오히려 수축한다(52.34 Å³).

> millerite는 Ni가 3개(홀수)라 보상된 collinear AFM이 불가능하다. 최소 2배 셀이 필요하고,
> R3m에서 3개 Ni가 대칭 등가라 collinear AFM은 기하학적으로 frustrated이다.
> 다만 millerite는 실험적으로 Pauli 상자성 금속이므로 NM이 실제 바닥상태이며,
> 각 상을 각자의 자성 바닥상태에 두는 현재 처리가 옳다.

---

## 8. 다형체 탐색 범위

`structures_all/`에 MP에서 27개를 받아 그중 9개를 계산했다. 미사용 18개는
MP 기준으로 전부 hull 위이며, 그 여유폭(원소 S 제외 시 최소 21 meV/atom)이
본 계산의 MP 대비 오차 {mad:.1f} meV/atom보다 훨씬 크다 — 추가해도 hull은 바뀌지 않는다.

| 조성 | 미사용 구조 (MP E_hull, meV/atom) |
|---|---|
| Ni (1) | hcp P6₃/mmc (45.8) |
| NiS₂ (3) | Pnnm (21.3), R-3m (35.0), Fd-3m (43.8) |
| S (14) | P2₁ (0.4), P2/c (0.9), P2 (6.3), Pnnm (10.3), P2₁/c (12.3), … R-3 (51.2) |

같은 조성에 구조를 둘 이상 계산한 것은 **NiS(2종)와 NiS₂(2종)**뿐이다.
MP에도 NiS는 mp-594(NiAs형)와 mp-1547(millerite) 둘뿐이므로,
"MP 전수와 일치한다"로 논거를 세우는 것이 안전하다.

---

## 9. 알려진 한계

- **U 스캔 40런 중 2건**이 완전 수렴이 아니다. `U6/Ni₉S₈`은 힘 기준을 −0.05 eV/Å로
  완화했고(진동 구간 에너지 폭 0.24 meV/atom이라 영향 없음), `U6/NiS₂ P2₁/c`는
  relax가 NSW를 소진했다(static 전자수렴은 정상). 후자는 U=6에서 hull 꼭짓점이
  아니므로(pyrite가 0.239 eV/atom 낮음) 보고 수치에 들어가지 않는다.
- U 스캔의 분해 산물은 FM 초기화다. 고 U에서 NiAs형만 AFM 이득을 받으므로
  서열 역전이 부분적으로 자성 처리의 인공물일 수 있다. 다만 millerite는 실험적으로
  모멘트가 없고 α-NiS는 있으므로 이 비대칭은 물리적으로도 타당하다.
- MP2020 보정은 실험 생성엔탈피에 fit된 경험적 값이다. 절대 Ef를 인용할 때는
  MP와 같은 척도라는 점을 명시하는 것이 안전하다.

---

## 10. 파일

```
~/tgm-work/hull/
├── NiS_formation_energy.md      이 문서
├── fig1_convex_hull.png         Ni-S convex hull
├── fig2_mp_validation.png       MP 대비 검증 (parity + 편차)
├── fig3_U_dependence.png        U 의존성 3분할
├── hull_final.tsv               결과 표 (15개 열)
├── hull_final.json              기계 판독용
├── hull_final_report.txt        전체 리포트 원문
├── hull_NiS_MP2020.png          서버에서 생성한 hull 그림
├── uscan/                       U 스캔 (리포트·그림·요약 JSON)
└── scripts/                     재현용 스크립트 전체
```

서버 재현:

```bash
ssh MASTER
cd ~/NiS_hull
bash scripts/status.sh                        # 9상 상태
~/venvs/pmg/bin/python scripts/hull_quick.py  # 표 + hull 그림
~/venvs/pmg/bin/python scripts/mu_test.py     # 기준물질 무관성 검증
~/venvs/pmg/bin/python scripts/analyze_uscan.py   # U 스캔
```

로컬 그림·문서 재생성:

```bash
cd ~/tgm-work/hull
python3 scripts/make_figures.py
python3 scripts/make_report.py
```

`~/venvs/pmg`는 pymatgen 2026.8.13 전용 venv (TGM에 새로 구축).
TGM 파티션 중 **g1·g2는 AVX가 없어** VASP 빌드가 `illegal instruction`으로 즉사한다 —
`g3,g4,g5,g6`만 사용할 것.
