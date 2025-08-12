import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class RoadProfile:
    """Single-hump road elevation profile with flat sides and a centered peak.

    Provides an external acceleration term derived from the road slope.
    Positive slope (uphill) yields negative external acceleration (resistance),
    negative slope (downhill) yields positive external acceleration (assist).
    """

    def __init__(
        self,
        total_length: float = 1500.0,
        num_points: int = 1000,
        baseline_y: float = 0.0,
        peak_height: float = 80.0,
        bump_width_fraction: float = 0.4,

    ) -> None:
        self.total_length = float(total_length)
        self.num_points = int(num_points)
        self.baseline_y = float(baseline_y)
        self.peak_height = float(peak_height)
        self.bump_width_fraction = float(bump_width_fraction)

        # Build x and y
        self.x = np.linspace(0.0, self.total_length, self.num_points)
        self.y = np.full_like(self.x, self.baseline_y)

        center_x = self.total_length / 2.0
        bump_width = self.total_length * self.bump_width_fraction
        start_x = center_x - bump_width / 2.0
        end_x = center_x + bump_width / 2.0

        mask = (self.x >= start_x) & (self.x <= end_x)
        # Raised-cosine bump (C1 continuous with zero slope at edges)
        self.y[mask] = self.baseline_y + self.peak_height * 0.5 * (
            1.0 - np.cos(2.0 * np.pi * (self.x[mask] - start_x) / (end_x - start_x))
        )

        # Use gradient to get slope with same length as x
        self.slope = np.gradient(self.y, self.x)
        # Map slope to external acceleration in [-slope_gain, slope_gain]
        self.slope_gain = 5.0





    def get_slope_map(self, ) -> np.ndarray:
        """Return the full slope array.

        returns tanh(slope) which lies in [-1, 1].
        """

        return np.tanh(self.slope)

    def _index_for_position(self, position: float) -> int:
        """Convert a longitudinal position to the nearest slope index."""
        clamped_pos = float(np.clip(position, 0.0, self.total_length))
        frac = clamped_pos / self.total_length if self.total_length > 0 else 0.0
        idx_float = frac * max(self.num_points - 1, 0)
        return int(np.clip(int(round(idx_float)), 0, max(self.num_points - 1, 0)))

    def get_external_acc(self, position: float) -> float:
        """External acceleration due to road slope at a given position.

        Positive slope (uphill) produces negative acceleration (resistance).
        Negative slope (downhill) produces positive acceleration (assist).
        The magnitude is limited to [-slope_gain, slope_gain].
        """
        if self.num_points <= 0:
            return 0.0
        idx = self._index_for_position(position)
        slope_unitless = np.tanh(self.slope[idx])  # in [-1, 1]
        # Uphill (positive slope) should resist forward motion
        return float(-self.slope_gain * slope_unitless)


    def plot(self) -> None:
        fig, ax1 = plt.subplots(figsize=(8, 3))
        ax1.plot(self.x, self.y, label="elevation (y)", color="tab:blue")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y", color="tab:blue")
        ax1.tick_params(axis='y', labelcolor='tab:blue')



        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Demo: plot road and external acceleration
    road = RoadProfile(total_length=1500.0, num_points=1000, peak_height=80.0, bump_width_fraction=0.4)
    road.plot(show_slope=True)









# a max = 4
# a min = -4

