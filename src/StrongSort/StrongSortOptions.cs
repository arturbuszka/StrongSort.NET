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
    /// (max_kalman_prediction_without_update): a track predicts while it has missed fewer frames than this.</summary>
    public int MaxKalmanPredictionsWithoutUpdate { get; init; } = 7;

    /// <summary>Use the chi-square Mahalanobis gate on the motion model during appearance matching.</summary>
    public bool EnableMotionGating { get; init; } = true;

    /// <summary>Noise model of the Kalman aspect-ratio component. <see cref="KalmanNoise.BpbReid"/> (tracklab
    /// bpbreid_strong_sort) scales it with the box height, so the gate effectively ignores pose-driven
    /// aspect changes; <see cref="KalmanNoise.StrongSort"/> keeps the tight constants of the original paper.</summary>
    public KalmanNoise KalmanNoise { get; init; } = KalmanNoise.BpbReid;

    /// <summary>Scale the measurement noise of the Mahalanobis gate by (1 - detection confidence), as in the
    /// Kalman update. Off in tracklab bpbreid_strong_sort, where the gate uses the unscaled noise.</summary>
    public bool GatingUsesConfidence { get; init; } = false;
}

/// <summary>Noise model of the aspect-ratio component of the Kalman filter.</summary>
public enum KalmanNoise
{
    /// <summary>Aspect-ratio noise proportional to the box height (tracklab bpbreid_strong_sort).</summary>
    BpbReid,

    /// <summary>Constant aspect-ratio noise (1e-2 / 1e-5 / 1e-1) of the original StrongSORT.</summary>
    StrongSort,
}
