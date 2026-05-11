** do not EDIT ** 
-- Keep it raw , word-for-word --

** **Minutes of the meeting [2026-05-10, Sun] **
1. Keep the encoder frozen; train only the probe head (linear/MLP evaluation protocol — Chen 2020 linear-eval, Caron 2021 DINO §A.2).
1.1) Then run a backbone-scaling comparison across ViT-G/14 and ViT-H/14 (model-size sweep, not an ablation since no component is removed).
2. Mitigate catastrophic forgetting via curriculum learning (Bengio 2009) at both levels:
- parameter curriculum: progressive layer unfreezing (surgical schedule, Lee ICLR'23)
- data curriculum: easy → hard sample ordering (e.g., low motion magnitude → high)
3. For the 8 motion-flow classes (post-filter cross-product of {still, slow, medium, fast} ×
{leftward, rightward, upward, downward}), generate:
3.1) per-class exemplar video clips (k-medoid or top-confidence representatives)
3.2) UMAP (McInnes 2018) projection of encoder embeddings, colored by motion-class label — to
inspect class separability in feature space (UMAP is dimensionality reduction, not clustering)