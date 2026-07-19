# StrongSort.Net

[![CI](https://github.com/arturbuszka/StrongSort.NET/actions/workflows/ci.yml/badge.svg)](https://github.com/arturbuszka/StrongSort.NET/actions/workflows/ci.yml)
[![NuGet](https://img.shields.io/nuget/v/StrongSort.Net.svg)](https://www.nuget.org/packages/StrongSort.Net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![.NET](https://img.shields.io/badge/.NET-8.0%20%7C%2010.0-512BD4.svg)](#)

Pure C# implementation of **StrongSORT** (["StrongSORT: Make DeepSORT Great Again"](https://arxiv.org/abs/2202.13514)) —
a multi-object tracker combining an NSA Kalman filter, EMA-smoothed appearance embeddings
and Hungarian assignment. No native or ML dependencies: bring your own detector and
(optionally) ReID embeddings.

This is an independent, clean-room implementation written from the published algorithm; it is
not affiliated with the original authors.

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

## Build & test

```
dotnet build StrongSort.slnx -c Release
dotnet test StrongSort.slnx -c Release
```

Targets `net8.0` and `net10.0`.

## License & attribution

MIT — see [LICENSE](LICENSE). The StrongSORT and DeepSORT algorithm references, and a note on the
clean-room nature of this port, are in [NOTICE](NOTICE).
