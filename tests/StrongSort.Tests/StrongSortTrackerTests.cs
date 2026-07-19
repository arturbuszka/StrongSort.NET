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
