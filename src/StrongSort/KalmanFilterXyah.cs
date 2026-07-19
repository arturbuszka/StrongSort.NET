namespace StrongSort;

/// <summary>
/// The 8-dimensional constant-velocity Kalman filter of DeepSORT/StrongSORT.
/// State: (x, y, a, h, vx, vy, va, vh) where (x, y) is the bbox center,
/// a the aspect ratio (w/h) and h the height. StrongSORT's NSA extension scales the
/// measurement noise by (1 - detection confidence).
/// </summary>
internal sealed class KalmanFilterXyah
{
    private const double StdWeightPosition = 1.0 / 20;
    private const double StdWeightVelocity = 1.0 / 160;

    /// <summary>0.95 quantile of the chi-square distribution, used as Mahalanobis gate.</summary>
    public static readonly double[] Chi2Inv95 = { 0, 3.8415, 5.9915, 7.8147, 9.4877 };

    public (double[] mean, double[,] covariance) Initiate(double[] measurement)
    {
        var mean = new double[8];
        Array.Copy(measurement, mean, 4);
        var h = measurement[3];
        var std = new[]
        {
            2 * StdWeightPosition * h, 2 * StdWeightPosition * h, 1e-2, 2 * StdWeightPosition * h,
            10 * StdWeightVelocity * h, 10 * StdWeightVelocity * h, 1e-5, 10 * StdWeightVelocity * h,
        };
        var covariance = new double[8, 8];
        for (var i = 0; i < 8; i++) covariance[i, i] = std[i] * std[i];
        return (mean, covariance);
    }

    public (double[] mean, double[,] covariance) Predict(double[] mean, double[,] covariance)
    {
        var h = mean[3];
        var motionStd = new[]
        {
            StdWeightPosition * h, StdWeightPosition * h, 1e-2, StdWeightPosition * h,
            StdWeightVelocity * h, StdWeightVelocity * h, 1e-5, StdWeightVelocity * h,
        };

        // x' = F x with F = [[I, I], [0, I]] (dt = 1)
        var newMean = new double[8];
        for (var i = 0; i < 4; i++) newMean[i] = mean[i] + mean[i + 4];
        for (var i = 4; i < 8; i++) newMean[i] = mean[i];

        // P' = F P F^T + Q
        var fp = new double[8, 8];
        for (var i = 0; i < 8; i++)
            for (var j = 0; j < 8; j++)
                fp[i, j] = covariance[i, j] + (i < 4 ? covariance[i + 4, j] : 0);
        var newCov = new double[8, 8];
        for (var i = 0; i < 8; i++)
            for (var j = 0; j < 8; j++)
                newCov[i, j] = fp[i, j] + (j < 4 ? fp[i, j + 4] : 0);
        for (var i = 0; i < 8; i++) newCov[i, i] += motionStd[i] * motionStd[i];
        return (newMean, newCov);
    }

    /// <summary>Projects state to measurement space; NSA: R scaled by (1 - confidence).</summary>
    public (double[] mean, double[,] covariance) Project(double[] mean, double[,] covariance, double confidence)
    {
        var h = mean[3];
        var std = new[] { StdWeightPosition * h, StdWeightPosition * h, 1e-1, StdWeightPosition * h };
        var nsa = 1 - confidence;

        var projMean = new double[4];
        Array.Copy(mean, projMean, 4);
        var projCov = new double[4, 4];
        for (var i = 0; i < 4; i++)
            for (var j = 0; j < 4; j++)
                projCov[i, j] = covariance[i, j];
        for (var i = 0; i < 4; i++)
            projCov[i, i] += std[i] * nsa * (std[i] * nsa);
        return (projMean, projCov);
    }

    public (double[] mean, double[,] covariance) Update(
        double[] mean, double[,] covariance, double[] measurement, double confidence)
    {
        var (projMean, projCov) = Project(mean, covariance, confidence);
        var projCovInv = Invert4x4(projCov);

        // Kalman gain K = P H^T S^-1 (H selects the first 4 state dims)
        var gain = new double[8, 4];
        for (var i = 0; i < 8; i++)
            for (var j = 0; j < 4; j++)
                for (var k = 0; k < 4; k++)
                    gain[i, j] += covariance[i, k] * projCovInv[k, j];

        var innovation = new double[4];
        for (var i = 0; i < 4; i++) innovation[i] = measurement[i] - projMean[i];

        var newMean = (double[])mean.Clone();
        for (var i = 0; i < 8; i++)
            for (var j = 0; j < 4; j++)
                newMean[i] += gain[i, j] * innovation[j];

        // P' = P - K S K^T
        var ks = new double[8, 4];
        for (var i = 0; i < 8; i++)
            for (var j = 0; j < 4; j++)
                for (var k = 0; k < 4; k++)
                    ks[i, j] += gain[i, k] * projCov[k, j];
        var newCov = new double[8, 8];
        for (var i = 0; i < 8; i++)
            for (var j = 0; j < 8; j++)
            {
                var kskT = 0.0;
                for (var k = 0; k < 4; k++) kskT += ks[i, k] * gain[j, k];
                newCov[i, j] = covariance[i, j] - kskT;
            }
        return (newMean, newCov);
    }

    /// <summary>Squared Mahalanobis distance between the state and measurements (gating).</summary>
    public double GatingDistance(double[] mean, double[,] covariance, double[] measurement, double confidence)
    {
        var (projMean, projCov) = Project(mean, covariance, confidence);
        var inv = Invert4x4(projCov);
        var d = new double[4];
        for (var i = 0; i < 4; i++) d[i] = measurement[i] - projMean[i];
        var result = 0.0;
        for (var i = 0; i < 4; i++)
            for (var j = 0; j < 4; j++)
                result += d[i] * inv[i, j] * d[j];
        return result;
    }

    private static double[,] Invert4x4(double[,] m)
    {
        // Gauss-Jordan with partial pivoting on an augmented [m | I] matrix.
        var a = new double[4, 8];
        for (var i = 0; i < 4; i++)
        {
            for (var j = 0; j < 4; j++) a[i, j] = m[i, j];
            a[i, i + 4] = 1;
        }
        for (var col = 0; col < 4; col++)
        {
            var pivot = col;
            for (var r = col + 1; r < 4; r++)
                if (Math.Abs(a[r, col]) > Math.Abs(a[pivot, col])) pivot = r;
            if (pivot != col)
                for (var j = 0; j < 8; j++)
                    (a[col, j], a[pivot, j]) = (a[pivot, j], a[col, j]);
            var diag = a[col, col];
            for (var j = 0; j < 8; j++) a[col, j] /= diag;
            for (var r = 0; r < 4; r++)
            {
                if (r == col) continue;
                var factor = a[r, col];
                for (var j = 0; j < 8; j++) a[r, j] -= factor * a[col, j];
            }
        }
        var inv = new double[4, 4];
        for (var i = 0; i < 4; i++)
            for (var j = 0; j < 4; j++)
                inv[i, j] = a[i, j + 4];
        return inv;
    }
}
