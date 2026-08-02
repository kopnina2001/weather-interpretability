# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 09 — Matrix/tensor decompositions for head & layer redundancy: storm vs quiet
#
# Follow-up to notebooks 07 (hidden-state/attention skeleton decomposition) and 08 (input-level
# climatological patching). Here we ask a different question: **do different attention heads
# (and layers) do redundant work, and does that change when the model is given a real storm vs
# a climatologically "blank" quiet case?** Three methods, in increasing sophistication:
#
# 1. **Singular value spectrum / effective rank** of the hidden state (plain SVD, extends the
#    skeleton-decomposition rank-vs-error sweep from notebook 07 into a full spectrum view).
# 2. **CKA (Centered Kernel Alignment)** between attention heads' *output representations* —
#    standard interpretability tool (Kornblith et al. 2019) for comparing whether two heads
#    encode the same information, independent of basis/scale.
# 3. **Tucker decomposition** of the (heads, query, key) attention *weight* tensor — a stricter,
#    numerical test of how many head-mode components are needed to reconstruct the raw attention
#    patterns (not just their downstream representation).
#
# All three are run on the same case pair as notebooks 07/08: Bebinca storm (init=2024-09-15,
# +24h through the 2024-09-16 landfall) vs a quiet control (init=2024-11-15). Method 1 reuses the
# raw hidden-state matrices already cached in notebook 07 (no rerun); methods 2/3 required a
# small new hook-based capture (window covering the Bebinca landfall area, block 0/non-shifted,
# stages 0 and 1 of the backbone).

# %% [markdown]
# ## 1. Singular value spectrum / effective rank of the hidden state

# %% [markdown]
# ![](legacy/figures/09_tensor_decomposition/spectrum_storm_vs_quiet.png)
#

# %% [markdown]
# Cumulative energy (fraction of total squared Frobenius norm) vs rank, at +24h, per backbone
# stage. **Negative result**: the storm and quiet curves are nearly indistinguishable at every
# stage (effective rank for 90% energy: stage 0 ~20-28, stage 1 ~238-278, stage 2 ~359-427,
# storm and quiet within a few percent of each other at every stage). If anything, storm's
# stage-0 spectrum is *slightly more* concentrated (lower rank) than quiet's, the opposite of the
# naive "storm injects more information" hypothesis.
#
# **Conclusion**: the hidden state's effective rank/compressibility is a structural property of
# backbone depth, not something driven by the presence of a real physical event. This complements
# (and slightly tempers) the notebook 07 finding that skeleton landmarks cluster at the storm
# core -- the *location* of informative structure shifts with the storm, but the overall
# *amount* of exploitable low-rank structure does not.

# %% [markdown]
# ## 2. CKA between attention heads (output representations)

# %% [markdown]
# ![](legacy/figures/09_tensor_decomposition/cka_heads_matrix.png)
#

# %% [markdown]
# Mean off-diagonal CKA (higher = heads more redundant/similar to each other):
#
# | | stage 0 | stage 1 |
# |---|---|---|
# | storm | 0.740 | 0.500 |
# | quiet | 0.886 | 0.532 |
#
# **Positive result**: heads are *more redundant* with each other in the quiet case than in the
# storm case, at both stages (bigger gap at stage 0). Visually, the quiet stage-0 matrix is
# almost uniformly high-similarity (yellow/green), while storm's has more spread. Head 5 (stage
# 0) is a consistent outlier in *both* cases -- a structural specialization independent of the
# storm, not something the storm creates.
#
# **Interpretation**: without a real storm to differentiate, the model's heads converge toward
# overlapping/redundant behavior; a real storm appears to pull heads toward more distinct roles
# -- consistent with the model needing a richer, less redundant internal representation to
# handle the more complex real dynamics.

# %% [markdown]
# ## 3. Tucker decomposition of the (heads, query, key) attention tensor

# %% [markdown]
# ![](legacy/figures/09_tensor_decomposition/tucker_head_rank.png)
#

# %% [markdown]
# Head-mode rank needed for a given Tucker reconstruction fidelity of the raw attention
# *weights* (a stricter numerical test than CKA, which compares representations up to
# rotation/scale). At stage 0, the storm curve sits consistently *above* the quiet curve --
# more head-rank is needed for the same reconstruction error, confirming the CKA finding with an
# independent, stricter method. At stage 1 the two curves nearly overlap -- the effect is
# concentrated at the finest resolution stage, not present (or much weaker) at the coarser stage.
#
# Rank needed for <10% relative error: storm stage0 = 8/8 (no compression possible at this
# tolerance), quiet stage0 = 7/8; stage1 both = 14/16. Tucker is a much stricter test than CKA
# (exact numerical reconstruction of softmax attention values vs correlation of representations),
# so the fact that the *direction* of the storm-vs-quiet effect agrees between the two very
# different methods is a stronger claim than either alone.

# %% [markdown]
# ## Summary (tasks 1-4)
#
# - **Effective rank** (method 1) of the hidden state is essentially unaffected by the storm --
#   compressibility is architectural, not event-driven.
# - **Head redundancy** (methods 2 CKA and 3 Tucker, two independent techniques) tells a
#   consistent, opposite story: the real storm makes heads *less* redundant with each other
#   (more specialized), at least at the finest backbone stage.
# - **RepE-style difference PCA** (method 4) was informative mainly as a negative/cautionary
#   result: comparing two different calendar dates conflates storm signal with seasonal signal.
#   Redoing this with the notebook 08 patching infrastructure (same date, real vs
#   climatology-patched) instead of two different dates would isolate the storm concept much
#   more cleanly.
# - **NMF** (method 5) on attention weights found a cleaner additive structure (column/zonal
#   attention stripes) than skeleton decomposition did on the same object, but the structure was
#   shared between storm and quiet -- an architectural property, not storm-specific. It did
#   surface a different "odd one out" head (head 2) than CKA did (head 5), suggesting stage-0
#   heads vary along more than one axis (what they attend to vs. what representation they
#   produce).
# - Task 5 from the original plan (sparse autoencoder / monosemantic features) remains as a
#   follow-up -- the highest-cost, highest-risk item, since it needs substantially more data
#   than a single storm + quiet case pair to reliably train interpretable features.

# %% [markdown]
# ## 4. RepE-style PCA on the storm-minus-quiet hidden-state difference
#
# Representation Engineering (Zou et al. 2023) decomposes the *difference* between two runs'
# activations (rather than one run in isolation) to find "concept directions". We did this on
# the same cached hidden-state matrices from notebook 07: `X_storm - X_quiet` per stage/lead
# time, then PCA of that difference matrix, with the spatial map of each component's score.

# %% [markdown]
# ![](legacy/figures/09_tensor_decomposition/repe_pca_scores.png)
#

# %% [markdown]
# **Caveat that matters**: PC1 at stage 0 (38% of the difference's variance) is a broad
# north-south gradient spanning the whole crop -- not localized at the storm at all. This looks
# like a **seasonal background difference** (Sept 15 vs Nov 15 are different seasons with
# different large-scale circulation), not an isolated typhoon signal. PC2 (18%) shows more
# local east-west structure but still isn't cleanly storm-centered the way the notebook 07
# skeleton landmarks were.
#
# **Honest conclusion**: a single storm-date-vs-single-quiet-date difference conflates "storm
# signal" with "seasonal signal" -- RepE normally uses *many* contrastive pairs controlled for
# everything except the target concept. Doing this properly would need either (a) many
# storm/no-storm date pairs *within the same season*, or (b) reusing the notebook 08
# patching infrastructure (same date, real vs climatology-patched input) instead of two
# different real dates, which would isolate the "storm" difference much more cleanly than
# comparing two different calendar dates ever can.

# %% [markdown]
# ## 5. Non-negative Matrix Factorization (NMF) of attention weights

# %% [markdown]
# ![](legacy/figures/09_tensor_decomposition/nmf_basis_patterns.png)
#

# %% [markdown]
# NMF (Lee & Seung 1999) as a "parts-based" alternative to skeleton decomposition, applied to
# the same object that gave a negative result in notebook 07 Section 5 (window attention
# matrices) -- but NMF needs no artificial data transform here, since softmax attention weights
# are already non-negative. k=6 basis patterns, shared across all heads and the 144 window
# positions.
#
# **What we found**: each NMF basis pattern is a clean vertical-stripe structure -- "attend to
# this one column (a specific relative-longitude offset), regardless of row" -- a much more
# interpretable additive structure than skeleton's edge/corner-selection artifact. The same
# stripe structure appears in **both** storm and quiet, though, so this looks like a general
# architectural bias in how this windowed attention organizes information (plausibly favoring
# zonal/longitudinal attention), not something storm-specific. Reconstruction error is still
# fairly high (0.69-0.93 with only k=6 components -- attention entropy is high, few components
# are not enough for a tight fit), so treat the basis patterns as qualitative structure, not a
# precise decomposition.

# %% [markdown]
# ![](legacy/figures/09_tensor_decomposition/nmf_head_loadings.png)
#

# %% [markdown]
# Per-head mean loading on each of the 6 components. **Head 2** (stage 0) is consistently the
# lowest-loading head across every component, in both storm and quiet -- its attention pattern
# isn't well described by the shared basis at all, i.e. it's doing something structurally
# different from the other 7 heads. This is a *different* outlier head from the one CKA found
# (head 5, notebook 09 Section 2) -- not a contradiction: CKA compared per-head *output
# representations* (attn @ v), NMF here decomposes the *attention weights* themselves, so
# "structurally distinct" means different things in each analysis. Together they suggest stage-0
# heads are not uniform along either axis (what they compute, or how they route).
