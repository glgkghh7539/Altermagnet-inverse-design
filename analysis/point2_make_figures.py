#!/usr/bin/env python3
"""Figures for reviewer Point 2 — Robustness of the SSE label.

Typography follows the paper's existing figures: Times New Roman (or a
metric-compatible clone when unavailable) with Computer Modern math.

Colour uses a validated categorical palette, assigned in fixed order and never
cycled. Forms that compare all pairs at once (scatter, small multiples) stay
within the first three slots.
"""
import os, csv, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
FIG  = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)

# --- palette (light mode) ---
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
SURFACE, INK, INK_2, INK_MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8880', '#e3e2dd'

plt.rcParams['font.family'] = ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK_2,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'text.color': INK, 'axes.linewidth': 0.8,
    'grid.color': GRID, 'grid.linewidth': 0.6,
    'legend.frameon': False, 'figure.dpi': 160,
})

def style(ax, title=None, xlabel=None, ylabel=None, grid='y'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if grid: ax.grid(axis=grid, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    if title:  ax.set_title(title, color=INK, pad=8, loc='left', fontweight='bold')
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

def rd(p):
    """The data are split into raw/derived/archive (reorganised 2026-08-22)."""
    for sub in ('raw', 'derived', '', '../archive'):
        f = os.path.join(DATA, sub, p) if sub else os.path.join(DATA, p)
        if os.path.isfile(f):
            return list(csv.DictReader(open(f)))
    raise IOError('not found: %s' % p)
def F(x):
    try: return float(x)
    except (TypeError, ValueError): return None


# ============================================================
# Fig 1. Orbital-character matching: m-resolved vs l-resolved
# ============================================================
om  = rd('orbital_match_all.csv')
sse = {r['name']: r for r in rd('sse_variants_all.csv')}
# raw is immutable and still contains the six removed rows; use only rows present in the current dataset.
om  = [r for r in om if r['name'] in sse]
cl = [F(r['cos_l_sum']) for r in om if F(r['cos_l_sum']) is not None]
cm = [F(r['cos_m_sum']) for r in om if F(r['cos_m_sum']) is not None]
hi_l = [F(r['cos_l_sum']) for r in om
        if F(r['cos_l_sum']) is not None and F(sse[r['name']]['sse_max']) >= 0.5]

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
bins = np.linspace(0, 1, 51)
for ax, vals, col, ttl in [(axes[0], cm, C[1], '$m$-resolved (fixed orbital basis)'),
                           (axes[1], cl, C[0], r'$\ell$-resolved (rotation-invariant)')]:
    ax.hist(vals, bins=bins, color=col, edgecolor=SURFACE, linewidth=0.4)
    med = np.median(vals)
    ax.axvline(med, color=INK, lw=1.2, ls='--')
    ax.text(med - 0.03, ax.get_ylim()[1]*0.92, 'median %.3f' % med,
            ha='right', color=INK, fontsize=9)
    style(ax, ttl, 'Cosine similarity', 'Number of calculations' if ax is axes[0] else None)
fig.suptitle('Spin-up / spin-down orbital character at the SSE maximum (3,851 calculations)',
             x=0.02, ha='left', fontweight='bold', color=INK)
fig.text(0.02, -0.05,
         'In an altermagnet the two spin sublattices are related by a rotation, so a mismatch of the '
         r'$m$-resolved components is expected by definition.' '\n'
         r'Measured with the rotation-invariant $\ell$-resolved vector the median is %.3f overall and '
         '%.3f for the high-SSE subset ($\\geq$ 0.5 eV).' % (np.median(cl), np.median(hi_l)),
         fontsize=9, color=INK_2)
fig.tight_layout(rect=[0, 0.02, 1, 0.93])
fig.savefig(os.path.join(FIG, 'fig1_orbital_matching.png'), bbox_inches='tight')
plt.close(fig)

# ============================================================
# Fig 2. Rank-matching diagnosis
# ============================================================
ta = [r for r in rd('typeA_all.csv') if r['name'] in sse]
n_ok = sum(1 for r in ta if r['type'] == 'rank_ok')
n_un = sum(1 for r in ta if r['direction'] == 'rank_under')
n_ov = sum(1 for r in ta if r['direction'] == 'rank_over')
tot = len(ta)
fig, ax = plt.subplots(figsize=(7.6, 2.2))
left = 0
for val, col in [(n_ok, C[0]), (n_un, C[2]), (n_ov, C[1])]:
    ax.barh([0], [val], left=left, color=col, height=0.5, edgecolor=SURFACE, linewidth=2)
    ax.text(left + val/2, 0, '%d\n%.1f%%' % (val, 100*val/tot), ha='center', va='center',
            color='white' if val > tot*0.06 else INK, fontsize=10, fontweight='bold')
    left += val
ax.set_xlim(0, tot); ax.set_yticks([])
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.set_xlabel('Number of calculations (total %d)' % tot)
ax.set_title(r'Rank-matching diagnosis — proximity constraint $|\Delta b| \leq 2$, $\Delta\cos > 0.10$',
             loc='left', color=INK, fontweight='bold', pad=8)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C[0], C[2], C[1])]
ax.legend(handles,
          ['rank_ok (rank pair is the best character match)',
           'Type A — rank underestimates', 'Type A — rank overestimates'],
          loc='upper center', bbox_to_anchor=(0.5, -0.38), ncol=3, fontsize=8.5)
fig.text(0.0, -0.34,
         'Of the 216 Type-A cases (5.6%), two thirds run in the underestimating direction: '
         'rank matching is conservative more often than not.', fontsize=9, color=INK_2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig2_rank_diagnosis.png'), bbox_inches='tight')
plt.close(fig)

# ============================================================
# Fig 3. Fragility of the maximum, and companion metrics
# ============================================================
rows = [r for r in rd('sse_variants_all.csv') if r['status'] == 'ok']
ratio = [F(r['sse_max'])/F(r['sse_p95']) for r in rows if F(r['sse_p95']) and F(r['sse_p95']) > 1e-9]
mx  = np.array([F(r['sse_max']) for r in rows])
p95 = np.array([F(r['sse_p95']) for r in rows])
bz  = np.array([F(r['sse_mean_w']) for r in rows])

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
axes[0].hist(ratio, bins=np.linspace(1, 5, 60), color=C[0], edgecolor=SURFACE, linewidth=0.4)
axes[0].axvline(np.median(ratio), color=INK, lw=1.2, ls='--')
axes[0].text(np.median(ratio) + 0.08, axes[0].get_ylim()[1]*0.9,
             'median %.2f' % np.median(ratio), color=INK, fontsize=9)
style(axes[0], 'The maximum sits far from the bulk',
      r'SSE$_{\mathrm{max}}$ / SSE$_{\mathrm{P95}}$', 'Number of calculations')
axes[1].scatter(mx, p95, s=5, alpha=0.35, color=C[0], linewidths=0, label='P95')
axes[1].scatter(mx, bz,  s=5, alpha=0.35, color=C[1], linewidths=0, label='BZ-weighted mean')
axes[1].plot([0, 1.9], [0, 1.9], color=INK_MUTED, lw=1, ls=':')
axes[1].set_xlim(0, 1.9); axes[1].set_ylim(0, 1.3)
style(axes[1], 'Companion metrics track the maximum',
      r'SSE$_{\mathrm{max}}$ (eV)', 'Companion metric (eV)', grid='both')
axes[1].legend(loc='upper left', fontsize=9)
fig.suptitle('Fragility of the maximum and its companion metrics (3,851 calculations)',
             x=0.02, ha='left', fontweight='bold', color=INK)
fig.text(0.02, -0.04,
         r'Median SSE$_{\mathrm{max}}$/SSE$_{\mathrm{P95}}$ = 1.80; 31.9% exceed 2. '
         r'Spearman $\rho$ = 0.976 (P95) and 0.968 (BZ mean).', fontsize=9, color=INK_2)
fig.tight_layout(rect=[0, 0.02, 1, 0.93])
fig.savefig(os.path.join(FIG, 'fig3_max_fragility.png'), bbox_inches='tight')
plt.close(fig)

# ============================================================
# Fig 4. k-point convergence (small multiples)
# ============================================================
conv = [r for r in rd('convergence_126runs.csv') if r['status'] == 'ok']
cv = {}
for r in conv:
    s, c = r['name'].split('|')
    cv.setdefault(s, {})[c] = r
MULT = [('base', r'$\times$1'), ('k24', r'$\times$1.33'), ('k133', r'$\times$1.33'),
        ('k27', r'$\times$1.5'), ('k150', r'$\times$1.5'),
        ('k36', r'$\times$2'), ('k200', r'$\times$2')]
SLOTS = [r'$\times$1', r'$\times$1.33', r'$\times$1.5', r'$\times$2']
ORDER = [('FeAs', 'candidate'), ('FeS', 'candidate'), ('CoS', 'candidate'),
         ('CoO_sp', 'candidate'), ('NiS', 'candidate'), ('CrS', 'candidate'),
         ('lowFeSi', 'low SSE'), ('lowBFe', 'low SSE'), ('lowAgF', 'low SSE'),
         ('sparseCrPb', 'low SSE')]
panels = [(s, k) for s, k in ORDER if s in cv and 'base' in cv[s]]
ncol = 3; nrow = int(math.ceil(len(panels)/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(10.0, 2.7*nrow))
axes = np.atleast_1d(axes).ravel()
for ax, (s, kind) in zip(axes, panels):
    ys = [np.nan]*len(SLOTS); nks = ['']*len(SLOTS)
    for c, m in MULT:
        if c in cv[s]:
            k = SLOTS.index(m)
            ys[k] = F(cv[s][c]['sse_max'])*1000
            nks[k] = cv[s][c]['nkpts']
    ys = np.array(ys, dtype=float); ok = ~np.isnan(ys)
    col = C[0] if kind == 'candidate' else C[1]
    x = np.arange(len(SLOTS))
    ax.plot(x[ok], ys[ok], '-o', color=col, lw=2, ms=7,
            markeredgecolor=SURFACE, markeredgewidth=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(['%s\n%s' % (SLOTS[i], nks[i] or '–') for i in range(len(SLOTS))],
                       fontsize=8)
    style(ax, '%s  (%s)' % (s.replace('_', r'\_') if False else s, kind), None,
          r'SSE$_{\mathrm{max}}$ (meV)')
    if ok.sum() >= 2:
        yv = ys[ok]
        rising = yv[-1] > yv[0]
        ax.text(0.97, 0.06 if rising else 0.90,
                'range %.2f meV' % (np.nanmax(ys) - np.nanmin(ys)),
                transform=ax.transAxes, ha='right',
                va='bottom' if rising else 'top', fontsize=9, color=INK_2)
    else:
        ax.text(0.97, 0.90, 'k-axis in progress', transform=ax.transAxes,
                ha='right', va='top', fontsize=9, color=INK_MUTED, style='italic')
    if kind == 'low SSE':
        ax.axhline(1.0, color=C[7], lw=1.3, ls='--')
        ax.text(0.02, 1.0, '1 meV', color=C[7], fontsize=8, va='bottom', ha='left',
                transform=ax.get_yaxis_transform())
        ax.set_ylim(bottom=0)
for ax in axes[len(panels):]: ax.set_visible(False)
fig.suptitle('Convergence of SSE$_{\\mathrm{max}}$ with k-point density   '
             '(blue = candidate, orange = low SSE;  numbers under the axis are NKPTS)',
             x=0.02, ha='left', fontweight='bold', color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(os.path.join(FIG, 'fig4_kpoint_convergence.png'), bbox_inches='tight')
plt.close(fig)

# ============================================================
# Fig 5. Non-k axes
# ============================================================
AX = [('Cutoff', ['enc600', 'enc625', 'enc700']),
      ('Smearing', ['sig0005', 'sig002', 'sig005', 'tetra']),
      ('Electronic tolerance', ['ediff1e7', 'ediff1e8'])]
syss = [s for s, _ in ORDER if s in cv and 'base' in cv[s]]
fig, ax = plt.subplots(figsize=(9.4, 3.7))
w = 0.26
for i, (aname, cs) in enumerate(AX):
    vals = []
    for s in syss:
        b = F(cv[s]['base']['sse_max'])
        v = [abs(F(cv[s][c]['sse_max']) - b)*1000 for c in cs if c in cv[s]]
        vals.append(max(v) if v else np.nan)
    ax.bar(np.arange(len(syss)) + (i-1)*w, vals, width=w*0.92, color=C[i],
           label=aname, edgecolor=SURFACE, linewidth=1)
ax.axhline(1.0, color=INK, lw=1.2, ls='--')
ax.text(len(syss)-0.4, 1.15, '1 meV inclusion threshold', ha='right', fontsize=9, color=INK)
ax.set_yscale('log')
ax.set_xticks(np.arange(len(syss))); ax.set_xticklabels(syss, rotation=20, ha='right')
style(ax, r'Largest $|\Delta$SSE$_{\mathrm{max}}|$ over cutoff, smearing and electronic tolerance',
      None, 'meV (log scale)')
ax.set_ylim(top=90)
ax.legend(loc='upper left', fontsize=9, ncol=3)
fig.text(0.02, -0.13,
         'The absolute shift never exceeds 30 meV on any axis, but what that means depends '
         'entirely on the size of the splitting. For the high-SSE\ncandidates it is at most 2.2% '
         '(FeS) and usually far below 0.1%. For the low-SSE systems the same 10–15 meV is a '
         '300–500% error, and\nlowAgF and lowBFe cross the inclusion threshold outright. FeS is '
         'the one high-SSE exception: its base run stopped at EDIFF = 1E-06\nbefore convergence, '
         'so the 28.7 meV shift is a tolerance artefact, not a physical sensitivity (see report '
         'section 6.6.1).',
         fontsize=9, color=INK_2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig5_nonk_axes.png'), bbox_inches='tight')
plt.close(fig)

# ============================================================
# Fig 6. The 1 meV inclusion threshold
# ============================================================
nk = np.array([int(r['nkpts']) for r in rows if r['nkpts']])
ss = np.array([F(r['sse_max']) for r in rows if r['nkpts']])
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))

# -- left: the four probe systems, showing NKPTS does NOT predict risk
PROBE = [('sparseCrPb', 86,  14.42, 15.81, False),
         ('lowAgF',     260,  0.56, 26.09, True),
         ('lowFeSi',    677,  2.67,  3.04, False),
         ('lowBFe',     788,  3.00,  6.56, False)]
ypos = np.arange(len(PROBE))
for i, (nm, n, lo, hi, cross) in enumerate(PROBE):
    col = C[7] if cross else C[2]
    axes[0].plot([lo, hi], [i, i], '-', color=col, lw=5, solid_capstyle='round', alpha=0.85)
    axes[0].plot([lo, hi], [i, i], 'o', color=col, ms=6,
                 markeredgecolor=SURFACE, markeredgewidth=1.2)
    axes[0].text(hi*1.25, i, '  NKPTS %d' % n, va='center', fontsize=8.5, color=INK_2)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels([p[0] for p in PROBE])
axes[0].set_xscale('log'); axes[0].set_xlim(0.3, 260)
axes[0].set_ylim(-0.75, len(PROBE)+0.35)
axes[0].axvline(1.0, color=C[7], lw=1.4, ls='--')
axes[0].text(1.06, -0.62, '1 meV', color=C[7], fontsize=8.5, va='center')
style(axes[0], 'k-point density does not predict the risk',
      r'SSE$_{\mathrm{max}}$ range over the k-axis (meV, log)', None)
axes[0].text(0.035, 0.98,
             'sparseCrPb is 3' + r'$\times$' + ' sparser than lowAgF\nyet far more stable',
             transform=axes[0].transAxes, ha='left', va='top',
             fontsize=8.5, color=INK_2, style='italic',
             bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.5))

# -- right: exposure as a function of the SSE cut alone
axes[1].scatter(nk, ss*1000, s=4, alpha=0.22, color=INK_MUTED, linewidths=0)
m_risk = ss < 0.010
axes[1].scatter(nk[m_risk], ss[m_risk]*1000, s=7, alpha=0.6, color=C[1], linewidths=0)
axes[1].set_xscale('log'); axes[1].set_yscale('log')
axes[1].axhline(1.0, color=C[7], lw=1.4, ls='--')
axes[1].axhline(10.0, color=C[1], lw=1.3, ls='-.')
axes[1].text(nk.max()*0.95, 1.0, '1 meV inclusion threshold', color=C[7],
             fontsize=8.5, va='center', ha='right',
             bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.5))
axes[1].text(nk.max()*0.95, 10.0, '10 meV — proposed cut', color=C[1],
             fontsize=8.5, va='center', ha='right',
             bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.5))
style(axes[1], 'Exposure is set by the SSE value, not by NKPTS', 'NKPTS (log)',
      r'SSE$_{\mathrm{max}}$ (meV, log)', grid='both')
n10 = int((ss < 0.010).sum()); n5 = int((ss < 0.005).sum()); N = len(ss)
axes[1].text(0.035, 0.97,
             'SSE < 10 meV : %d  (%.1f%%)\nSSE < 5 meV : %d  (%.1f%%)'
             % (n10, 100*n10/N, n5, 100*n5/N),
             transform=axes[1].transAxes, ha='left', va='top', color=C[1],
             fontsize=9.5, fontweight='bold')
fig.suptitle('How far the 1 meV inclusion threshold can be trusted',
             x=0.02, ha='left', fontweight='bold', color=INK)
fig.text(0.02, -0.10,
         'The earlier criterion (NKPTS < 500 and SSE < 30 meV, 695 entries) is not supported by '
         'the data: sparseCrPb sits inside it at NKPTS 86\nand is stable to 1%, while lowAgF at '
         'NKPTS 260 swings by a factor of 46. What separates them is the size of the splitting, '
         'not the mesh.\nOn an SSE-only criterion the real exposure is 153 entries (4.0%) below '
         '10 meV, of which 24 (0.6%) lie below 5 meV.',
         fontsize=9, color=INK_2)
fig.tight_layout(rect=[0, 0.02, 1, 0.93])
fig.savefig(os.path.join(FIG, 'fig6_threshold_1meV.png'), bbox_inches='tight')
plt.close(fig)

# ============================================================
# Fig 7. Candidate ranking across SSE definitions
# ============================================================
cand = {r['name']: r for r in rd('candidates_sse.csv') if r['status'] == 'ok'}
DEFS = [('sse_max', 'max'), ('sse_p99', 'P99'), ('sse_p95', 'P95'),
        ('sse_p90', 'P90'), ('sse_mean_w', 'BZ mean'), ('sse_kmax_mean_w', 'mean of per-k max')]
names = sorted(cand, key=lambda n: -F(cand[n]['sse_max']))
rank = {n: [] for n in names}
for k, _ in DEFS:
    for i, n in enumerate(sorted(names, key=lambda m: -F(cand[m][k]))):
        rank[n].append(i+1)
fig, ax = plt.subplots(figsize=(8.6, 3.9))
x = np.arange(len(DEFS))
for i, n in enumerate(names):
    ax.plot(x, rank[n], '-o', color=C[i], lw=2, ms=7,
            markeredgecolor=SURFACE, markeredgewidth=1.4, label=n)
    ax.annotate(n, (x[-1], rank[n][-1]), xytext=(8, 0), textcoords='offset points',
                color=C[i], fontsize=9.5, va='center', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([d[1] for d in DEFS])
ax.set_yticks(range(1, len(names)+1)); ax.invert_yaxis()
ax.set_xlim(-0.3, len(DEFS) - 0.3 + 1.3)
style(ax, 'Candidate ranking across six SSE definitions', 'SSE definition', 'Rank (1 = highest)')
fig.text(0.02, -0.05,
         'FeAs and FeS hold the top two places under every definition and CrS stays sixth; '
         'only the middle three exchange places between ranks 3 and 5.', fontsize=9, color=INK_2)
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(os.path.join(FIG, 'fig7_candidate_ranking.png'), bbox_inches='tight')
plt.close(fig)

# ============================================================
# Fig 8. SHAP ranking across retraining targets
# ============================================================
TGT = [('sse_orig', 'SSE max\n(published)'), ('sse_max_new', 'SSE max\n(re-extracted)'),
       ('p95', 'P95'), ('bzmean', 'BZ mean')]
KEY = [('p_metric', 'MSBI'), ('packing_fraction', 'MPF'), ('pd_ratio', 'p/d ratio')]
sh = {t: {r['feature']: int(r['rank']) for r in rd('shap200f_%s.csv' % t)} for t, _ in TGT}
fig, ax = plt.subplots(figsize=(8.2, 3.5))
w = 0.26
for i, (f, lab) in enumerate(KEY):
    vals = [sh[t][f] for t, _ in TGT]
    xs = np.arange(len(TGT)) + (i-1)*w
    ax.bar(xs, vals, width=w*0.92, color=C[i], label=lab, edgecolor=SURFACE, linewidth=1)
    for xx, v in zip(xs, vals):
        ax.text(xx, v + 0.12, str(v), ha='center', va='bottom', fontsize=9.5,
                color=INK, fontweight='bold')
ax.set_xticks(np.arange(len(TGT))); ax.set_xticklabels([t[1] for t in TGT])
ax.set_ylim(0, 8); ax.invert_yaxis()
style(ax, 'SHAP importance rank when the training target changes (out of 52 features)',
      'Training target', 'Rank (1 = most important)')
ax.legend(loc='lower right', fontsize=9, ncol=3)
fig.text(0.02, -0.06,
         'MSBI is first under all four targets, and stays first across four '
         'independent runs (16/16). MPF and p/d hold 2nd and 3rd-4th here; '
         'under the P95 target the ranks below 1st move between runs. '
         'GroupKFold out-of-fold evaluation, Optuna, 200 trials.', fontsize=9, color=INK_2)
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(os.path.join(FIG, 'fig8_shap_ranking.png'), bbox_inches='tight')
plt.close(fig)
print('all figures written to', os.path.normpath(FIG))
