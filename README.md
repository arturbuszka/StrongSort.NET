# StrongSort.Net

[![CI](https://github.com/arturbuszka/StrongSort.NET/actions/workflows/ci.yml/badge.svg)](https://github.com/arturbuszka/StrongSort.NET/actions/workflows/ci.yml)
[![NuGet](https://img.shields.io/nuget/v/StrongSort.Net.svg)](https://www.nuget.org/packages/StrongSort.Net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![.NET](https://img.shields.io/badge/.NET-8.0%20%7C%2010.0-512BD4.svg)](#)

Pure C# implementation of **StrongSORT** (["StrongSORT: Make DeepSORT Great Again"](https://arxiv.org/abs/2202.13514)) —
a multi-object tracker combining an NSA Kalman filter, EMA-smoothed appearance embeddings
and Hungarian assignment. No native or ML dependencies: bring your own detector and
(optionally) ReID embeddings.

This is an independent implementation, not affiliated with the original authors. It was written
from the published algorithm, and since 2.0.0 its Kalman noise, gating and camera motion update are
aligned with the `bpbreid_strong_sort` tracker in [TrackLab](https://github.com/TrackingLaboratory/tracklab).

## Install

```
dotnet add package StrongSort.Net
```

## Usage

```csharp
using StrongSort;

var tracker = new StrongSortTracker(new StrongSortOptions
{
    MaxAppearanceDistance = 0.5f,
    MaxAge = 300,
});

foreach (var frame in video)
{
    var observations = detector.Detect(frame)
        .Select(d => new Observation(d.Left, d.Top, d.Width, d.Height, d.Confidence,
                                     Feature: reid.Embed(frame, d), Tag: d.Id));
    IReadOnlyList<TrackResult> tracks = tracker.Update(observations.ToList());
    // tracks: stable TrackId, Kalman-smoothed BboxLtwh, Tag of the matched observation
}
```

- Works without embeddings too (`Feature: null`) — matching then falls back to the IoU stage.
- `Tag` is an arbitrary payload (e.g. your detection id) returned on the matched track.
- Defaults follow the SoccerNet game-state baseline; original paper defaults are documented
  on `StrongSortOptions`.
- `Update(observations, cameraMotion)` takes an optional 3x3 matrix that moves every track with the
  camera after the prediction. Leave it out when the camera is static or you have no estimate.

## Upgrading from 1.x

2.0.0 changes default tracking results, so expect different track IDs on the same video:

- `KalmanNoise` defaults to `BpbReid`, which scales aspect-ratio noise with box height.
- `GatingUsesConfidence` defaults to `false`, so the motion gate uses the unscaled measurement noise.
- A track now predicts while it has missed fewer than `MaxKalmanPredictionsWithoutUpdate` frames.
  It used to include that last frame too.

To approximate 1.x results, set `KalmanNoise = KalmanNoise.StrongSort`,
`GatingUsesConfidence = true` and raise `MaxKalmanPredictionsWithoutUpdate` by one (7 becomes 8).

`Update()` gained an optional parameter, so code built against 1.x must be recompiled.

## Build & test

```
dotnet build StrongSort.slnx -c Release
dotnet test StrongSort.slnx -c Release
```

Targets `net8.0` and `net10.0`.

## License & attribution

MIT — see [LICENSE](LICENSE). The StrongSORT and DeepSORT algorithm references and the TrackLab
attribution are in [NOTICE](NOTICE).
