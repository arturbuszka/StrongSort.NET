namespace StrongSort;

/// <summary>
/// StrongSORT multi-object tracker: per-frame association of detections to tracks using
/// a blend of appearance (cosine distance on EMA-smoothed ReID embeddings) and motion
/// (NSA Kalman filter with Mahalanobis gating), followed by an IoU matching stage.
/// Feed one frame at a time via <see cref="Update"/>.
/// </summary>
public sealed class StrongSortTracker
{
    private readonly StrongSortOptions _options;
    private readonly KalmanFilterXyah _kf = new();
    private readonly List<Track> _tracks = new();
    private long _nextId = 1;

    /// <summary>Creates a tracker; omit <paramref name="options"/> for the SoccerNet baseline defaults.</summary>
    public StrongSortTracker(StrongSortOptions? options = null)
        => _options = options ?? new StrongSortOptions();

    /// <summary>Drops all tracks and resets id numbering (call between videos).</summary>
    public void Reset()
    {
        _tracks.Clear();
        _nextId = 1;
    }

    /// <summary>Advances the tracker by one frame and returns all live tracks.
    /// Tracks matched in this frame carry the observation's Tag and TimeSinceUpdate == 0.</summary>
    public IReadOnlyList<TrackResult> Update(IReadOnlyList<Observation> observations)
    {
        var detections = observations
            .Where(o => o.Confidence >= _options.MinConfidence)
            .Select(o => o with { Feature = Normalize(o.Feature) })
            .ToList();

        foreach (var track in _tracks)
            track.Predict(_kf, _options.MaxKalmanPredictionsWithoutUpdate);

        var (matches, unmatchedTracks, unmatchedDetections) = Match(detections);

        foreach (var (trackIdx, detIdx) in matches)
            _tracks[trackIdx].Update(_kf, detections[detIdx], _options);
        foreach (var trackIdx in unmatchedTracks)
            _tracks[trackIdx].MarkMissed(_options.MaxAge);
        foreach (var detIdx in unmatchedDetections)
            _tracks.Add(new Track(_nextId++, _kf, detections[detIdx], _options.NInit));
        _tracks.RemoveAll(t => t.State == TrackState.Deleted);

        return _tracks
            .Where(t => t.State == TrackState.Confirmed)
            .Select(t => t.ToResult())
            .ToList();
    }

    private (List<(int Track, int Det)> matches, List<int> unmatchedTracks, List<int> unmatchedDets)
        Match(List<Observation> detections)
    {
        var confirmed = Range(_tracks.Count).Where(i => _tracks[i].State == TrackState.Confirmed).ToList();
        var unconfirmed = Range(_tracks.Count).Where(i => _tracks[i].State != TrackState.Confirmed).ToList();

        // Stage 1: appearance + gated motion on confirmed tracks.
        var (matchesA, unmatchedTracksA, unmatchedDets) =
            MinCostMatching(BuildAppearanceMotionCost, confirmed, Range(detections.Count).ToList(), detections);

        // Stage 2 (as in DeepSORT/StrongSORT): IoU matching for unconfirmed tracks plus
        // confirmed tracks that were missed for exactly one frame.
        var iouTrackCandidates = unconfirmed
            .Concat(unmatchedTracksA.Where(i => _tracks[i].TimeSinceUpdate == 1))
            .ToList();
        var remainingTracks = unmatchedTracksA.Where(i => _tracks[i].TimeSinceUpdate != 1).ToList();

        var (matchesB, unmatchedTracksB, unmatchedDetsB) =
            MinCostMatching(BuildIouCost, iouTrackCandidates, unmatchedDets, detections);

        return (matchesA.Concat(matchesB).ToList(),
                remainingTracks.Concat(unmatchedTracksB).ToList(),
                unmatchedDetsB);
    }

    private float[,] BuildAppearanceMotionCost(List<int> trackIndices, List<int> detIndices, List<Observation> detections)
    {
        var gate = KalmanFilterXyah.Chi2Inv95[4];
        var cost = new float[trackIndices.Count, detIndices.Count];
        for (var r = 0; r < trackIndices.Count; r++)
        {
            var track = _tracks[trackIndices[r]];
            for (var c = 0; c < detIndices.Count; c++)
            {
                var det = detections[detIndices[c]];
                var appearance = CosineDistance(track.Feature, det.Feature);
                if (appearance > _options.MaxAppearanceDistance)
                {
                    cost[r, c] = float.PositiveInfinity;
                    continue;
                }
                var mahalanobis = _kf.GatingDistance(track.Mean, track.Covariance, ToXyah(det), det.Confidence);
                if (_options.EnableMotionGating && mahalanobis > gate)
                {
                    cost[r, c] = float.PositiveInfinity;
                    continue;
                }
                var motion = (float)Math.Min(mahalanobis / gate, 1.0);
                cost[r, c] = _options.McLambda * appearance + (1 - _options.McLambda) * motion;
            }
        }
        return cost;
    }

    private float[,] BuildIouCost(List<int> trackIndices, List<int> detIndices, List<Observation> detections)
    {
        var cost = new float[trackIndices.Count, detIndices.Count];
        for (var r = 0; r < trackIndices.Count; r++)
        {
            var trackBox = _tracks[trackIndices[r]].BboxLtwh();
            for (var c = 0; c < detIndices.Count; c++)
            {
                var det = detections[detIndices[c]];
                var iouCost = 1 - Iou(trackBox, new[] { det.Left, det.Top, det.Width, det.Height });
                cost[r, c] = iouCost > _options.MaxIouDistance ? float.PositiveInfinity : iouCost;
            }
        }
        return cost;
    }

    private (List<(int, int)>, List<int>, List<int>) MinCostMatching(
        Func<List<int>, List<int>, List<Observation>, float[,]> costFn,
        List<int> trackIndices, List<int> detIndices, List<Observation> detections)
    {
        if (trackIndices.Count == 0 || detIndices.Count == 0)
            return (new List<(int, int)>(), new List<int>(trackIndices), new List<int>(detIndices));

        var cost = costFn(trackIndices, detIndices, detections);
        var assignment = HungarianSolver.Solve(cost);

        var matches = new List<(int, int)>();
        var unmatchedTracks = new List<int>();
        var matchedDets = new HashSet<int>();
        for (var r = 0; r < trackIndices.Count; r++)
        {
            if (assignment[r] >= 0)
            {
                matches.Add((trackIndices[r], detIndices[assignment[r]]));
                matchedDets.Add(assignment[r]);
            }
            else
            {
                unmatchedTracks.Add(trackIndices[r]);
            }
        }
        var unmatchedDets = Range(detIndices.Count).Where(c => !matchedDets.Contains(c))
                                                   .Select(c => detIndices[c]).ToList();
        return (matches, unmatchedTracks, unmatchedDets);
    }

    internal static double[] ToXyah(Observation o)
        => new double[] { o.Left + o.Width / 2, o.Top + o.Height / 2, o.Width / Math.Max(o.Height, 1e-6f), o.Height };

    private static float[]? Normalize(float[]? feature)
    {
        if (feature is null) return null;
        var norm = MathF.Sqrt(feature.Sum(x => x * x));
        if (norm < 1e-12f) return feature;
        return feature.Select(x => x / norm).ToArray();
    }

    /// <summary>Cosine distance of two L2-normalized vectors; 1 (neutral max) when either is missing.</summary>
    private static float CosineDistance(float[]? a, float[]? b)
    {
        if (a is null || b is null || a.Length != b.Length) return 1f;
        var dot = 0f;
        for (var i = 0; i < a.Length; i++) dot += a[i] * b[i];
        return 1 - dot;
    }

    private static float Iou(float[] a, float[] b)
    {
        var x1 = Math.Max(a[0], b[0]);
        var y1 = Math.Max(a[1], b[1]);
        var x2 = Math.Min(a[0] + a[2], b[0] + b[2]);
        var y2 = Math.Min(a[1] + a[3], b[1] + b[3]);
        var inter = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
        var union = a[2] * a[3] + b[2] * b[3] - inter;
        return union <= 0 ? 0 : inter / union;
    }

    private static IEnumerable<int> Range(int count) => Enumerable.Range(0, count);
}

/// <summary>Internal track state: Kalman state + EMA appearance feature.</summary>
internal sealed class Track
{
    public long Id { get; }
    public TrackState State { get; private set; }
    public double[] Mean { get; private set; }
    public double[,] Covariance { get; private set; }
    public float[]? Feature { get; private set; }
    public int TimeSinceUpdate { get; private set; }

    private int _hits = 1;
    private readonly int _nInit;
    private float _lastConfidence;
    private object? _lastTag;

    public Track(long id, KalmanFilterXyah kf, Observation observation, int nInit)
    {
        Id = id;
        _nInit = nInit;
        (Mean, Covariance) = kf.Initiate(StrongSortTracker.ToXyah(observation));
        Feature = observation.Feature;
        _lastConfidence = observation.Confidence;
        _lastTag = observation.Tag;
        State = nInit <= 1 ? TrackState.Confirmed : TrackState.Tentative;
    }

    public void Predict(KalmanFilterXyah kf, int maxPredictionsWithoutUpdate)
    {
        if (TimeSinceUpdate <= maxPredictionsWithoutUpdate)
            (Mean, Covariance) = kf.Predict(Mean, Covariance);
        TimeSinceUpdate++;
        _lastTag = null;
    }

    public void Update(KalmanFilterXyah kf, Observation observation, StrongSortOptions options)
    {
        (Mean, Covariance) = kf.Update(Mean, Covariance, StrongSortTracker.ToXyah(observation), observation.Confidence);
        if (observation.Feature is not null)
        {
            if (Feature is null)
            {
                Feature = observation.Feature;
            }
            else
            {
                var smoothed = new float[Feature.Length];
                var norm = 0f;
                for (var i = 0; i < Feature.Length; i++)
                {
                    smoothed[i] = options.EmaAlpha * Feature[i] + (1 - options.EmaAlpha) * observation.Feature[i];
                    norm += smoothed[i] * smoothed[i];
                }
                norm = MathF.Sqrt(norm);
                for (var i = 0; i < smoothed.Length; i++) smoothed[i] /= norm;
                Feature = smoothed;
            }
        }
        TimeSinceUpdate = 0;
        _lastConfidence = observation.Confidence;
        _lastTag = observation.Tag;
        _hits++;
        if (State == TrackState.Tentative && _hits >= _nInit)
            State = TrackState.Confirmed;
    }

    public void MarkMissed(int maxAge)
    {
        if (State == TrackState.Tentative || TimeSinceUpdate > maxAge)
            State = TrackState.Deleted;
    }

    public float[] BboxLtwh()
    {
        var w = (float)(Mean[2] * Mean[3]);
        var h = (float)Mean[3];
        return new[] { (float)Mean[0] - w / 2, (float)Mean[1] - h / 2, w, h };
    }

    public TrackResult ToResult() => new()
    {
        TrackId = Id,
        State = State,
        BboxLtwh = BboxLtwh(),
        Confidence = TimeSinceUpdate == 0 ? _lastConfidence : 0,
        TimeSinceUpdate = Math.Max(TimeSinceUpdate, 0),
        Tag = _lastTag,
    };
}
