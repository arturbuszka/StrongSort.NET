namespace StrongSort;

/// <summary>One detector output for the current frame, in image pixels.</summary>
/// <param name="Left">Left edge of the bbox.</param>
/// <param name="Top">Top edge of the bbox.</param>
/// <param name="Width">Bbox width.</param>
/// <param name="Height">Bbox height.</param>
/// <param name="Confidence">Detector confidence in [0, 1].</param>
/// <param name="Feature">Optional ReID appearance embedding (any fixed length; will be L2-normalized).</param>
/// <param name="Tag">Caller-owned payload returned on the matched track (e.g. a detection id).</param>
public readonly record struct Observation(
    float Left, float Top, float Width, float Height,
    float Confidence, float[]? Feature = null, object? Tag = null);

/// <summary>Lifecycle state of a track.</summary>
public enum TrackState
{
    /// <summary>Recently created; awaiting <c>NInit</c> consecutive hits before confirmation.</summary>
    Tentative,
    /// <summary>Established track, reported from <see cref="StrongSortTracker.Update"/>.</summary>
    Confirmed,
    /// <summary>Lost track scheduled for removal.</summary>
    Deleted,
}

/// <summary>A track as reported after <see cref="StrongSortTracker.Update"/> —
/// the Kalman-smoothed bbox plus the observation matched in this frame (if any).</summary>
public sealed class TrackResult
{
    /// <summary>Stable identity assigned by the tracker (1-based, unique within a tracker instance).</summary>
    public required long TrackId { get; init; }

    /// <summary>Lifecycle state; results only contain <see cref="TrackState.Confirmed"/> tracks.</summary>
    public required TrackState State { get; init; }

    /// <summary>Kalman-filtered bbox: left, top, width, height.</summary>
    public required float[] BboxLtwh { get; init; }

    /// <summary>Confidence of the matched observation (0 when coasting).</summary>
    public required float Confidence { get; init; }

    /// <summary>Frames since the last matched observation (0 = matched this frame).</summary>
    public required int TimeSinceUpdate { get; init; }

    /// <summary>Tag of the observation matched this frame, or null when coasting.</summary>
    public object? Tag { get; init; }
}
