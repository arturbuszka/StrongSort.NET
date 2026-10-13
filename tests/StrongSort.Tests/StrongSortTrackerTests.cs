using StrongSort;

namespace StrongSort.Tests;

public class StrongSortTrackerTests
{
    private static Observation Obs(float x, float y, float[]? feature = null, float conf = 0.9f, object? tag = null)
        => new(x, y, 40, 100, conf, feature, tag);

    private static float[] Feature(params float[] values) => values;

    [Fact]
    public void SingleObject_KeepsSameTrackIdAcrossFrames()
    {
        var tracker = new StrongSortTracker();
        var feature = Feature(1, 0, 0);

        long? id = null;
        for (var frame = 0; frame < 10; frame++)
        {
            var tracks = tracker.Update(new[] { Obs(100 + frame * 5, 200, feature, tag: frame) });
            var track = Assert.Single(tracks);
            id ??= track.TrackId;
            Assert.Equal(id, track.TrackId);
            Assert.Equal(frame, track.Tag);
        }
    }

    [Fact]
    public void TwoCrossingObjects_AppearanceKeepsIdentities()
    {
        var tracker = new StrongSortTracker();
        var featureA = Feature(1, 0, 0);
        var featureB = Feature(0, 1, 0);

        // A moves right, B moves left; they swap positions around frame 10.
        long idA = 0, idB = 0;
        for (var frame = 0; frame < 20; frame++)
        {
            var tracks = tracker.Update(new[]
            {
                Obs(100 + frame * 10, 200, featureA, tag: "A"),
                Obs(300 - frame * 10, 200, featureB, tag: "B"),
            });
            Assert.Equal(2, tracks.Count);
            var trackA = tracks.Single(t => Equals(t.Tag, "A"));
            var trackB = tracks.Single(t => Equals(t.Tag, "B"));
            if (frame == 0) (idA, idB) = (trackA.TrackId, trackB.TrackId);
            Assert.Equal(idA, trackA.TrackId);
            Assert.Equal(idB, trackB.TrackId);
        }
    }

    [Fact]
    public void OcclusionGap_ReassociatesByAppearance()
    {
        var tracker = new StrongSortTracker();
        var feature = Feature(0.6f, 0.8f, 0);

        var first = tracker.Update(new[] { Obs(100, 200, feature) });
        var originalId = Assert.Single(first).TrackId;

        for (var i = 0; i < 5; i++)
            tracker.Update(Array.Empty<Observation>()); // occlusion: no detections

        var after = tracker.Update(new[] { Obs(130, 200, feature, tag: "back") });
        var reacquired = after.Single(t => Equals(t.Tag, "back"));
        Assert.Equal(originalId, reacquired.TrackId);
    }

    private static readonly StrongSortOptions OriginalStrongSortGating = new()
    {
        KalmanNoise = KalmanNoise.StrongSort,
        GatingUsesConfidence = true,
    };

    // After a short gap (no IoU stage) the person returns in place with a different pose (wider box).
    private static long? ReturnWithNewPose(StrongSortOptions options)
    {
        var tracker = new StrongSortTracker(options);
        var feature = Feature(0.6f, 0.8f, 0);
        long id = 0;
        for (var frame = 0; frame < 5; frame++)
            id = Assert.Single(tracker.Update(new[] { Obs(100, 200, feature) })).TrackId;
        for (var i = 0; i < 3; i++)
            tracker.Update(Array.Empty<Observation>());

        var wide = new Observation(100, 200, 55, 100, 0.9f, feature, "back");
        var back = tracker.Update(new[] { wide }).Single(t => Equals(t.Tag, "back"));
        return back.TrackId == id ? id : null;
    }

    [Fact]
    public void PoseChange_AfterGap_KeepsIdentity_WithBpbReidKalman()
        => Assert.NotNull(ReturnWithNewPose(new StrongSortOptions()));

    [Fact]
    public void PoseChange_AfterGap_IsGatedOut_WithOriginalStrongSortNoise()
        => Assert.Null(ReturnWithNewPose(OriginalStrongSortGating));

    [Theory]
    [InlineData(KalmanNoise.BpbReid, false, true)]
    [InlineData(KalmanNoise.StrongSort, false, true)]
    [InlineData(KalmanNoise.StrongSort, true, false)]
    public void Gate_AspectChange(KalmanNoise noise, bool useConfidence, bool passes)
    {
        var kf = new KalmanFilterXyah(noise);
        var (mean, cov) = kf.Initiate(new double[] { 100, 200, 0.4, 100 });
        for (var i = 0; i < 5; i++)
        {
            (mean, cov) = kf.Predict(mean, cov);
            (mean, cov) = kf.Update(mean, cov, new double[] { 100, 200, 0.4, 100 }, 0.9);
        }
        (mean, cov) = kf.Predict(mean, cov);

        var distance = kf.GatingDistance(mean, cov, new double[] { 100, 200, 0.5, 100 }, useConfidence ? 0.9 : 0);
        Assert.Equal(passes, distance <= KalmanFilterXyah.Chi2Inv95[4]);
    }

    [Fact]
    public void KalmanPrediction_StopsAfterMaxPredictionsWithoutUpdate()
    {
        var tracker = new StrongSortTracker(new StrongSortOptions { MaxKalmanPredictionsWithoutUpdate = 3 });
        for (var frame = 0; frame < 10; frame++)
            tracker.Update(new[] { Obs(100 + frame * 10, 200) }); // builds up a velocity

        var lefts = new List<float>();
        for (var i = 0; i < 6; i++)
            lefts.Add(Assert.Single(tracker.Update(Array.Empty<Observation>())).BboxLtwh[0]);

        // misses 1..3 still move the box, from the 4th miss on it stays put
        Assert.All(Enumerable.Range(1, 2), i => Assert.True(lefts[i] > lefts[i - 1]));
        Assert.All(Enumerable.Range(3, 3), i => Assert.Equal(lefts[2], lefts[i]));
    }

    private static double[,] Shift(double dx, double dy) => new double[,] { { 1, 0, dx }, { 0, 1, dy }, { 0, 0, 1 } };

    [Fact]
    public void CameraMotion_MovesTrackBox()
    {
        var tracker = new StrongSortTracker();
        tracker.Update(new[] { Obs(100, 200) });
        var box = Assert.Single(tracker.Update(Array.Empty<Observation>(), Shift(-30, 5))).BboxLtwh;
        Assert.Equal(70, box[0], 3);
        Assert.Equal(205, box[1], 3);
        Assert.Equal(40, box[2], 3);
        Assert.Equal(100, box[3], 3);
    }

    // A standing person leaves the frame while the camera pans 10 px/frame, then is detected again
    // 40 frames later, 400 px further left in the image: far beyond the motion gate of the frozen prediction.
    private static bool ReturnsAfterPan(bool compensate)
    {
        var tracker = new StrongSortTracker();
        var feature = Feature(0.6f, 0.8f, 0);
        long id = 0;
        for (var frame = 0; frame < 5; frame++)
            id = Assert.Single(tracker.Update(new[] { Obs(800, 200, feature) })).TrackId;
        for (var i = 0; i < 40; i++)
            tracker.Update(Array.Empty<Observation>(), compensate ? Shift(-10, 0) : null);
        var back = tracker.Update(new[] { Obs(400 - 10, 200, feature, tag: "back") }, compensate ? Shift(-10, 0) : null)
            .Single(t => Equals(t.Tag, "back"));
        return back.TrackId == id;
    }

    [Fact]
    public void CameraMotion_KeepsIdentityAfterPan() => Assert.True(ReturnsAfterPan(compensate: true));

    [Fact]
    public void WithoutCameraMotion_PanBreaksIdentity() => Assert.False(ReturnsAfterPan(compensate: false));

    [Fact]
    public void NoFeatures_FallsBackToIouMatching()
    {
        var tracker = new StrongSortTracker();

        var first = tracker.Update(new[] { Obs(100, 200) });
        var id = Assert.Single(first).TrackId;

        var second = tracker.Update(new[] { Obs(105, 202, tag: "next") });
        var track = second.Single(t => Equals(t.Tag, "next"));
        Assert.Equal(id, track.TrackId);
    }

    [Fact]
    public void StaleTrack_IsDeletedAfterMaxAge()
    {
        var tracker = new StrongSortTracker(new StrongSortOptions { MaxAge = 3 });

        tracker.Update(new[] { Obs(100, 200) });
        IReadOnlyList<TrackResult> tracks = Array.Empty<TrackResult>();
        for (var i = 0; i < 6; i++)
            tracks = tracker.Update(Array.Empty<Observation>());
        Assert.Empty(tracks);

        // A new object afterwards gets a fresh id.
        var reborn = tracker.Update(new[] { Obs(100, 200) });
        Assert.Single(reborn);
    }

    [Fact]
    public void LowConfidenceDetections_AreIgnored()
    {
        var tracker = new StrongSortTracker(new StrongSortOptions { MinConfidence = 0.5f });
        var tracks = tracker.Update(new[] { Obs(100, 200, conf: 0.3f) });
        Assert.Empty(tracks);
    }

    [Fact]
    public void HungarianSolver_PicksGloballyOptimalAssignment()
    {
        var cost = new float[2, 2];
        cost[0, 0] = 1; cost[0, 1] = 2;
        cost[1, 0] = 2; cost[1, 1] = 10;
        // Greedy would pick (0,0)+(1,1)=11; optimal is (0,1)+(1,0)=4.
        var assignment = HungarianSolver.Solve(cost);
        Assert.Equal(1, assignment[0]);
        Assert.Equal(0, assignment[1]);
    }

    [Fact]
    public void HungarianSolver_RespectsForbiddenPairs()
    {
        var cost = new float[1, 2];
        cost[0, 0] = float.PositiveInfinity;
        cost[0, 1] = 5;
        var assignment = HungarianSolver.Solve(cost);
        Assert.Equal(1, assignment[0]);
    }
}
