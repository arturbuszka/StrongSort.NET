namespace StrongSort;

/// <summary>
/// Hyperparameters of the tracker. Defaults follow the SoccerNet game-state baseline
/// (tracklab bpbreid_strong_sort.yaml); the original StrongSORT paper defaults are noted inline.
/// </summary>
public sealed record StrongSortOptions
{
    /// <summary>EMA smoothing of the per-track appearance embedding:
    /// feature = alpha * old + (1 - alpha) * new.</summary>
    public float EmaAlpha { get; init; } = 0.9f;

    /// <summary>Blend of matching costs: mc_lambda * appearance + (1 - mc_lambda) * motion.</summary>
    public float McLambda { get; init; } = 0.995f;

    /// <summary>Appearance gate: cosine distances above this are an invalid match (max_dist; paper 0.2).</summary>
    public float MaxAppearanceDistance { get; init; } = 0.5f;

    /// <summary>IoU gate of the second matching stage: cost (1 - IoU) above this is disregarded (paper 0.7).</summary>
    public float MaxIouDistance { get; init; } = 0.8f;

    /// <summary>Frames a track survives without a match before deletion (paper 70).</summary>
    public int MaxAge { get; init; } = 300;

    /// <summary>Consecutive hits required before a track is confirmed; 0 confirms immediately (paper 3).</summary>
    public int NInit { get; init; } = 0;

    /// <summary>Detections below this confidence are ignored.</summary>
    public float MinConfidence { get; init; } = 0f;

    /// <summary>Stop advancing the Kalman prediction after this many frames without measurements
    /// (max_kalman_prediction_without_update).</summary>
    public int MaxKalmanPredictionsWithoutUpdate { get; init; } = 7;

    /// <summary>Use the chi-square Mahalanobis gate on the motion model during appearance matching.</summary>
    public bool EnableMotionGating { get; init; } = true;
}
