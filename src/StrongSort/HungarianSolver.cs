namespace StrongSort;

/// <summary>
/// Min-cost assignment (Hungarian algorithm, O(n^3) Jonker-Volgenant style shortest
/// augmenting paths). Rectangular matrices are handled by padding internally.
/// </summary>
internal static class HungarianSolver
{
    /// <summary>Returns for each row the assigned column, or -1 when unassigned.
    /// Costs of <see cref="float.PositiveInfinity"/> mark forbidden pairs.</summary>
    public static int[] Solve(float[,] cost)
    {
        var rows = cost.GetLength(0);
        var cols = cost.GetLength(1);
        var n = Math.Max(rows, cols);
        const double forbidden = 1e9;

        // Pad to square; forbidden/padded cells get a large finite cost so the
        // algorithm terminates, and are filtered out afterwards.
        var a = new double[n + 1, n + 1];
        for (var i = 1; i <= n; i++)
            for (var j = 1; j <= n; j++)
                a[i, j] = i <= rows && j <= cols && !float.IsInfinity(cost[i - 1, j - 1])
                    ? cost[i - 1, j - 1] : forbidden;

        var u = new double[n + 1];
        var v = new double[n + 1];
        var p = new int[n + 1]; // p[j] = row assigned to column j
        var way = new int[n + 1];

        for (var i = 1; i <= n; i++)
        {
            p[0] = i;
            var j0 = 0;
            var minv = new double[n + 1];
            var used = new bool[n + 1];
            Array.Fill(minv, double.MaxValue);
            do
            {
                used[j0] = true;
                int i0 = p[j0], j1 = 0;
                var delta = double.MaxValue;
                for (var j = 1; j <= n; j++)
                {
                    if (used[j]) continue;
                    var cur = a[i0, j] - u[i0] - v[j];
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
                for (var j = 0; j <= n; j++)
                {
                    if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                    else minv[j] -= delta;
                }
                j0 = j1;
            } while (p[j0] != 0);
            do
            {
                var j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0 != 0);
        }

        var result = new int[rows];
        Array.Fill(result, -1);
        for (var j = 1; j <= n; j++)
        {
            var i = p[j];
            if (i >= 1 && i <= rows && j <= cols && !float.IsInfinity(cost[i - 1, j - 1]))
                result[i - 1] = j - 1;
        }
        return result;
    }
}
