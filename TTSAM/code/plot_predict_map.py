import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics

class TaiwanIntensity:
    label = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
    pga = np.log10(
        [1e-5, 0.008, 0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0]
    )  # log10(m/s^2)
    pgv = np.log10(
        [1e-5, 0.002, 0.007, 0.019, 0.057, 0.15, 0.3, 0.5, 0.8, 1.4]
    )  # log10(m/s)

    def __init__(self):
        self.pga_ticks = self.get_ticks(self.pga)
        self.pgv_ticks = self.get_ticks(self.pgv)

    def calculate(self, pga, pgv=None, label=False):
        pga_intensity = bisect.bisect(self.pga, pga) - 1
        intensity = pga_intensity

        if pga > self.pga[5] and pgv is not None:
            pgv_intensity = bisect.bisect(self.pgv, pgv) - 1
            if pgv_intensity > pga_intensity:
                intensity = pgv_intensity

        if label:
            return self.label[intensity]
        else:
            return intensity

    @staticmethod
    def get_ticks(array):
        ticks = np.cumsum(array, dtype=float)
        ticks[2:] = ticks[2:] - ticks[:-2]
        ticks = ticks[1:] / 2
        ticks = np.append(ticks, (ticks[-1] * 2 - ticks[-2]))
        return ticks

def true_predicted(
    y_true,
    y_pred,
    time,
    agg="mean",
    quantile=True,
    ms=None,
    ax=None,
    axis_fontsize=20,
    point_size=2,
    target="pga",
    title=None,
):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
    ax.set_aspect("equal")

    if quantile:
        c_quantile = np.sum(
            y_pred[:, :, 0]
            * (
                1
                - norm.cdf((y_true.reshape(-1, 1) - y_pred[:, :, 1]) / y_pred[:, :, 2])
            ),
            axis=-1,
            keepdims=False,
        )
    else:
        c_quantile = None

    if agg == "mean":
        y_pred_point = np.sum(y_pred[:, :, 0] * y_pred[:, :, 1], axis=1)
    elif agg == "point":
        y_pred_point = y_pred
    else:
        raise ValueError(f'Aggregation type "{agg}" unknown')

    limits = (np.min(y_true) - 0.5, np.max(y_true) + 0.5)
    ax.plot(limits, limits, "k-", zorder=1)
    if ms is None:
        cbar = ax.scatter(
            y_true,
            y_pred_point,
            c=c_quantile,
            cmap="coolwarm",
            zorder=2,
            alpha=0.3,
            s=point_size,
        )
    else:
        cbar = ax.scatter(
            y_true, y_pred_point, s=ms, c=c_quantile, cmap="coolwarm", zorder=2
        )

    intensity = TaiwanIntensity()
    if target == "pga":
        intensity_threshold = intensity.pga
        ticks = intensity.pga_ticks
    elif target == "pgv":
        intensity_threshold = intensity.pgv
        ticks = intensity.pgv_ticks
    ax.hlines(
        intensity_threshold[3:-1],
        limits[0],
        intensity_threshold[3:-1],
        linestyles="dotted",
    )
    ax.vlines(
        intensity_threshold[3:-1],
        limits[0],
        intensity_threshold[3:-1],
        linestyles="dotted",
    )
    for i, label in zip(ticks[2:-2], intensity.label[2:-2]):
        ax.text(i, limits[0], label, va="bottom", fontsize=axis_fontsize - 7)

    ax.set_xlabel(r"True PGA log(${m/s^2}$)", fontsize=axis_fontsize)
    ax.set_ylabel(r"Predicted PGA log(${m/s^2}$)", fontsize=axis_fontsize)
    if title == None:
        ax.set_title(f"Model prediction", fontsize=axis_fontsize + 5)
    else:
        ax.set_title(title, fontsize=axis_fontsize + 5)
    ax.tick_params(axis="x", labelsize=axis_fontsize - 5)
    ax.tick_params(axis="y", labelsize=axis_fontsize - 5)
    # ax.set_ylim(-3.5,1.5)
    # ax.set_xlim(-3.5,1.5)

    r2 = metrics.r2_score(y_true, y_pred_point)
    ax.text(
        min(np.min(y_true), limits[0]),
        max(np.max(y_pred_point), limits[1]),
        f"$R^2={r2:.2f}$",
        va="top",
        fontsize=axis_fontsize - 5,
    )

    return fig, ax
