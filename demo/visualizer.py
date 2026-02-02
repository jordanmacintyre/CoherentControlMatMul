"""
Visualization utilities for the coherent control loop.

Provides real-time and post-hoc visualization of:
- Phase trajectories
- Error convergence
- Matrix evolution
- I/Q constellation diagrams
"""

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from typing import TYPE_CHECKING
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if TYPE_CHECKING:
    from .control_loop import CoherentController, ControlState
    from .svd_control_loop import SVDCoherentController, SVDControlState


class ControlVisualizer:
    """
    Visualizer for the coherent control loop.

    Provides various plots and animations for understanding
    the calibration process.
    """

    def __init__(self, figsize: tuple[int, int] = (14, 10)):
        """
        Initialize visualizer.

        Args:
            figsize: Figure size in inches
        """
        self.figsize = figsize
        self._fig = None
        self._axes = None

    def plot_convergence(
        self,
        error_history: list[float],
        title: str = "Calibration Convergence",
        threshold: float | None = None,
    ) -> plt.Figure:
        """
        Plot error convergence over iterations.

        Args:
            error_history: List of error values
            title: Plot title
            threshold: Optional threshold line

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        iterations = np.arange(len(error_history))
        ax.semilogy(iterations, error_history, "b-", linewidth=2, label="Error")

        if threshold is not None:
            ax.axhline(
                y=threshold, color="r", linestyle="--", linewidth=1.5, label="Threshold"
            )

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Error (log scale)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_phase_trajectories(
        self,
        phase_history: list[NDArray[np.float64]],
        title: str = "Phase Trajectories",
    ) -> plt.Figure:
        """
        Plot phase values over iterations.

        Args:
            phase_history: List of phase arrays
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        phase_history = np.array(phase_history)
        iterations = np.arange(len(phase_history))

        phase_names = ["θ (splitting)", "φ₀ (input 0)", "φ₁ (input 1)", "φ_out (global)"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for i, (ax, name, color) in enumerate(zip(axes, phase_names, colors)):
            ax.plot(iterations, phase_history[:, i], color=color, linewidth=2)
            ax.set_xlabel("Iteration", fontsize=10)
            ax.set_ylabel("Phase (rad)", fontsize=10)
            ax.set_title(f"{name}", fontsize=12)
            ax.grid(True, alpha=0.3)

            # Add horizontal lines at multiples of π
            for mult in range(-2, 5):
                ax.axhline(y=mult * np.pi, color="gray", linestyle=":", alpha=0.5)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig

    def plot_matrix_evolution(
        self,
        matrix_history: list[NDArray[np.complex128]],
        target: NDArray[np.float64],
        title: str = "Matrix Evolution",
    ) -> plt.Figure:
        """
        Plot evolution of matrix elements.

        Args:
            matrix_history: List of measured matrices
            target: Target matrix
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        matrix_history = np.array(matrix_history)
        iterations = np.arange(len(matrix_history))

        element_names = [["M₀₀", "M₀₁"], ["M₁₀", "M₁₁"]]

        for i in range(2):
            for j in range(2):
                ax = axes[i, j]

                # Plot real part
                real_part = np.real(matrix_history[:, i, j])
                ax.plot(
                    iterations,
                    real_part,
                    "b-",
                    linewidth=2,
                    label="Real (measured)",
                )

                # Plot imaginary part
                imag_part = np.imag(matrix_history[:, i, j])
                ax.plot(
                    iterations,
                    imag_part,
                    "g-",
                    linewidth=1.5,
                    alpha=0.7,
                    label="Imag (measured)",
                )

                # Plot target
                ax.axhline(
                    y=target[i, j],
                    color="r",
                    linestyle="--",
                    linewidth=2,
                    label="Target",
                )

                ax.set_xlabel("Iteration", fontsize=10)
                ax.set_ylabel("Value", fontsize=10)
                ax.set_title(element_names[i][j], fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                ax.set_ylim(-1.5, 1.5)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig

    def plot_matrix_heatmap(
        self,
        measured: NDArray[np.complex128],
        target: NDArray[np.float64],
        title: str = "Matrix Comparison",
    ) -> plt.Figure:
        """
        Plot heatmap comparison of measured vs target matrix.

        Args:
            measured: Measured complex matrix
            target: Target real matrix
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Target matrix
        im0 = axes[0].imshow(
            target, cmap="RdBu", vmin=-1, vmax=1, aspect="equal"
        )
        axes[0].set_title("Target Matrix", fontsize=12)
        axes[0].set_xticks([0, 1])
        axes[0].set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                axes[0].text(
                    j, i, f"{target[i, j]:.3f}", ha="center", va="center", fontsize=14
                )
        plt.colorbar(im0, ax=axes[0])

        # Measured matrix (real part)
        measured_real = np.real(measured)
        im1 = axes[1].imshow(
            measured_real, cmap="RdBu", vmin=-1, vmax=1, aspect="equal"
        )
        axes[1].set_title("Measured (Real Part)", fontsize=12)
        axes[1].set_xticks([0, 1])
        axes[1].set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                axes[1].text(
                    j,
                    i,
                    f"{measured_real[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=14,
                )
        plt.colorbar(im1, ax=axes[1])

        # Error matrix
        error = measured_real - target
        max_err = max(0.1, np.max(np.abs(error)))
        im2 = axes[2].imshow(
            error, cmap="RdBu", vmin=-max_err, vmax=max_err, aspect="equal"
        )
        axes[2].set_title("Error (Measured - Target)", fontsize=12)
        axes[2].set_xticks([0, 1])
        axes[2].set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                axes[2].text(
                    j, i, f"{error[i, j]:.3f}", ha="center", va="center", fontsize=14
                )
        plt.colorbar(im2, ax=axes[2])

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig

    def plot_iq_constellation(
        self,
        matrix_history: list[NDArray[np.complex128]],
        target: NDArray[np.float64],
        title: str = "I/Q Constellation",
    ) -> plt.Figure:
        """
        Plot I/Q constellation diagram showing matrix element evolution.

        Args:
            matrix_history: List of measured matrices
            target: Target real matrix
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        matrix_history = np.array(matrix_history)
        element_names = [["M₀₀", "M₀₁"], ["M₁₀", "M₁₁"]]
        colors = plt.cm.viridis(np.linspace(0, 1, len(matrix_history)))

        for i in range(2):
            for j in range(2):
                ax = axes[i, j]

                # Plot trajectory
                values = matrix_history[:, i, j]
                for k in range(len(values) - 1):
                    ax.plot(
                        [np.real(values[k]), np.real(values[k + 1])],
                        [np.imag(values[k]), np.imag(values[k + 1])],
                        color=colors[k],
                        alpha=0.5,
                        linewidth=1,
                    )

                # Mark start and end
                ax.scatter(
                    np.real(values[0]),
                    np.imag(values[0]),
                    color="green",
                    s=100,
                    zorder=5,
                    label="Start",
                    marker="o",
                )
                ax.scatter(
                    np.real(values[-1]),
                    np.imag(values[-1]),
                    color="red",
                    s=100,
                    zorder=5,
                    label="End",
                    marker="s",
                )

                # Mark target (on real axis)
                ax.scatter(
                    target[i, j],
                    0,
                    color="blue",
                    s=150,
                    zorder=6,
                    label="Target",
                    marker="*",
                )

                # Unit circle
                circle = Circle((0, 0), 1, fill=False, color="gray", linestyle="--")
                ax.add_patch(circle)

                ax.set_xlabel("In-phase (I)", fontsize=10)
                ax.set_ylabel("Quadrature (Q)", fontsize=10)
                ax.set_title(element_names[i][j], fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                ax.set_aspect("equal")
                ax.legend(fontsize=8)
                ax.axhline(y=0, color="gray", linewidth=0.5)
                ax.axvline(x=0, color="gray", linewidth=0.5)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig

    def plot_summary(
        self,
        controller: "CoherentController",
        title: str = "Calibration Summary",
    ) -> plt.Figure:
        """
        Create a comprehensive summary plot.

        Args:
            controller: Controller with completed calibration
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(16, 12))

        # Layout: 2x3 grid
        # [convergence | phases_0 | phases_1]
        # [matrix_heat | iq_0     | iq_1    ]

        state = controller.state
        target = controller.target_matrix

        # 1. Convergence plot
        ax1 = fig.add_subplot(2, 3, 1)
        iterations = np.arange(len(state.error_history))
        ax1.semilogy(iterations, state.error_history, "b-", linewidth=2)
        ax1.axhline(
            y=controller.config.error_threshold,
            color="r",
            linestyle="--",
            label="Threshold",
        )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Error (log)")
        ax1.set_title("Convergence")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2-3. Phase trajectories
        phase_history = np.array(state.phase_history)
        for idx, (start, end) in enumerate([(0, 2), (2, 4)]):
            ax = fig.add_subplot(2, 3, idx + 2)
            for i in range(start, end):
                ax.plot(
                    iterations,
                    phase_history[:, i],
                    linewidth=2,
                    label=f"φ{i}",
                )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Phase (rad)")
            ax.set_title(f"Phase Trajectories {start}-{end-1}")
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 4. Matrix heatmap
        ax4 = fig.add_subplot(2, 3, 4)
        measured = controller.measure_matrix()
        measured_real = np.real(measured)
        im = ax4.imshow(measured_real, cmap="RdBu", vmin=-1, vmax=1)
        ax4.set_title("Final Matrix (Real)")
        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                ax4.text(
                    j,
                    i,
                    f"{measured_real[i, j]:.2f}\n(t:{target[i, j]:.2f})",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
        plt.colorbar(im, ax=ax4)

        # 5-6. I/Q constellations for diagonal elements
        matrix_history = np.array(state.matrix_history)
        for idx, (i, j) in enumerate([(0, 0), (1, 1)]):
            ax = fig.add_subplot(2, 3, idx + 5)

            values = matrix_history[:, i, j]
            colors = plt.cm.viridis(np.linspace(0, 1, len(values)))

            for k in range(len(values) - 1):
                ax.plot(
                    [np.real(values[k]), np.real(values[k + 1])],
                    [np.imag(values[k]), np.imag(values[k + 1])],
                    color=colors[k],
                    alpha=0.5,
                )

            ax.scatter(np.real(values[0]), np.imag(values[0]), color="g", s=80, zorder=5)
            ax.scatter(np.real(values[-1]), np.imag(values[-1]), color="r", s=80, zorder=5)
            ax.scatter(target[i, j], 0, color="b", s=120, marker="*", zorder=6)

            ax.set_xlabel("I")
            ax.set_ylabel("Q")
            ax.set_title(f"M{i}{j} I/Q Trajectory")
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="gray", linewidth=0.5)
            ax.axvline(x=0, color="gray", linewidth=0.5)

        # Add overall title with results
        status = "LOCKED" if state.locked else "NOT LOCKED"
        plt.suptitle(
            f"{title}\n{status} after {state.iteration} iterations | "
            f"Final error: {state.error:.2e}",
            fontsize=14,
        )

        plt.tight_layout()
        return fig

    def plot_svd_summary(
        self,
        controller: "SVDCoherentController",
        title: str = "SVD Calibration Summary",
    ) -> plt.Figure:
        """
        Create a comprehensive summary plot for SVD calibration.

        Args:
            controller: SVD controller with completed calibration
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(16, 14))

        # Layout: 3x3 grid
        # [convergence | V phases   | U phases  ]
        # [sigma_bar   | matrix     | error     ]
        # [iq_00       | iq_11      | svd_info  ]

        state = controller.state
        target = controller.target_matrix
        U_target, sigma_target, Vh_target = controller.target_svd

        # 1. Convergence plot
        ax1 = fig.add_subplot(3, 3, 1)
        if state.error_history:
            iterations = np.arange(len(state.error_history))
            ax1.semilogy(iterations, state.error_history, "b-", linewidth=2)
            ax1.axhline(
                y=controller.config.error_threshold,
                color="r",
                linestyle="--",
                label="Threshold",
            )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Error (log)")
        ax1.set_title("Convergence")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. V† phase trajectories
        ax2 = fig.add_subplot(3, 3, 2)
        if state.phase_history_v:
            phase_history_v = np.array(state.phase_history_v)
            iters_v = np.arange(len(phase_history_v))
            for i in range(4):
                ax2.plot(iters_v, phase_history_v[:, i], linewidth=2, label=f"φ_v{i}")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Phase (rad)")
        ax2.set_title("V† Mesh Phases")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)

        # 3. U phase trajectories
        ax3 = fig.add_subplot(3, 3, 3)
        if state.phase_history_u:
            phase_history_u = np.array(state.phase_history_u)
            iters_u = np.arange(len(phase_history_u))
            for i in range(4):
                ax3.plot(iters_u, phase_history_u[:, i], linewidth=2, label=f"φ_u{i}")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Phase (rad)")
        ax3.set_title("U Mesh Phases")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)

        # 4. Singular values bar chart
        ax4 = fig.add_subplot(3, 3, 4)
        x_pos = np.array([0, 1])
        width = 0.35
        # Target singular values
        ax4.bar(x_pos - width/2, sigma_target, width, label="Target σ", color="blue", alpha=0.7)
        # Achieved singular values (from VOA DAC codes)
        sigma_achieved = controller.plant.voas.dac_to_sigma(state.sigma_dac)
        ax4.bar(x_pos + width/2, sigma_achieved, width, label="Achieved σ", color="green", alpha=0.7)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(["σ₀", "σ₁"])
        ax4.set_ylabel("Singular Value")
        ax4.set_title("Singular Values (Σ)")
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis="y")

        # 5. Final matrix heatmap
        ax5 = fig.add_subplot(3, 3, 5)
        measured = controller.measure_matrix()
        measured_real = np.real(measured)
        im = ax5.imshow(measured_real, cmap="RdBu", vmin=-1, vmax=1)
        ax5.set_title("Final Matrix (Real)")
        ax5.set_xticks([0, 1])
        ax5.set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                ax5.text(
                    j,
                    i,
                    f"{measured_real[i, j]:.2f}\n(t:{target[i, j]:.2f})",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
        plt.colorbar(im, ax=ax5)

        # 6. Error matrix
        ax6 = fig.add_subplot(3, 3, 6)
        error_matrix = measured_real - target
        max_err = max(0.1, np.max(np.abs(error_matrix)))
        im6 = ax6.imshow(error_matrix, cmap="RdBu", vmin=-max_err, vmax=max_err)
        ax6.set_title("Error (Measured - Target)")
        ax6.set_xticks([0, 1])
        ax6.set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                ax6.text(
                    j, i, f"{error_matrix[i, j]:.3f}",
                    ha="center", va="center", fontsize=10
                )
        plt.colorbar(im6, ax=ax6)

        # 7. All 4 matrix elements I/Q trajectory on one plot
        ax7 = fig.add_subplot(3, 3, 7)
        if state.matrix_history:
            matrix_history = np.array(state.matrix_history)
            element_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Blue, orange, green, red
            element_labels = ["M00", "M01", "M10", "M11"]

            for idx, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                values = matrix_history[:, i, j]
                color = element_colors[idx]

                # Plot trajectory as scatter with fading alpha (newer = darker)
                n_points = len(values)
                alphas = np.linspace(0.2, 1.0, n_points)
                sizes = np.linspace(10, 40, n_points)

                for k in range(n_points):
                    ax7.scatter(
                        np.real(values[k]), np.imag(values[k]),
                        color=color, alpha=alphas[k], s=sizes[k],
                        marker="o" if k < n_points - 1 else "s"
                    )

                # Final position with label
                ax7.scatter(
                    np.real(values[-1]), np.imag(values[-1]),
                    color=color, s=80, marker="s", edgecolors="black",
                    linewidth=1.5, label=f"{element_labels[idx]}: {np.real(values[-1]):.2f}",
                    zorder=10
                )

                # Target position
                ax7.scatter(
                    target[i, j], 0, color=color, s=100, marker="*",
                    edgecolors="black", linewidth=0.5, zorder=11
                )

            ax7.set_xlabel("Real (I)")
            ax7.set_ylabel("Imag (Q)")
            ax7.set_title("Matrix Elements I/Q (stars = targets)")
            ax7.set_xlim(-1.5, 1.5)
            ax7.set_ylim(-0.5, 0.5)
            ax7.set_aspect("equal")
            ax7.grid(True, alpha=0.3)
            ax7.axhline(y=0, color="gray", linewidth=0.5)
            ax7.axvline(x=0, color="gray", linewidth=0.5)
            ax7.legend(fontsize=7, loc="upper right")
        else:
            ax7.text(0.5, 0.5, "No trajectory data", ha="center", va="center", transform=ax7.transAxes)
            ax7.set_title("Matrix Elements I/Q")

        # 8. Matrix element values over time (real part) with target lines
        ax8 = fig.add_subplot(3, 3, 8)
        if state.matrix_history:
            matrix_history = np.array(state.matrix_history)
            element_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
            element_labels = ["M00", "M01", "M10", "M11"]

            for idx, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                values = matrix_history[:, i, j]
                # Plot real part of measured values
                ax8.plot(np.real(values), color=element_colors[idx], linewidth=1.5,
                         label=f"{element_labels[idx]}: {target[i,j]:.2f}")
                # Target as horizontal dashed line
                ax8.axhline(y=target[i, j], color=element_colors[idx], linestyle="--",
                           linewidth=1, alpha=0.7)

            ax8.set_xlabel("Iteration")
            ax8.set_ylabel("Matrix Element Value")
            ax8.set_title("Element Values vs Targets (dashed)")
            ax8.grid(True, alpha=0.3)
            ax8.legend(fontsize=7, loc="best")
        else:
            ax8.text(0.5, 0.5, "No trajectory data", ha="center", va="center", transform=ax8.transAxes)
            ax8.set_title("Element Values Over Time")

        # 9. SVD info text
        ax9 = fig.add_subplot(3, 3, 9)
        ax9.axis("off")

        info_text = (
            f"SVD Architecture: M = U · Σ · V†\n\n"
            f"Target Singular Values:\n"
            f"  σ₀ = {sigma_target[0]:.4f}\n"
            f"  σ₁ = {sigma_target[1]:.4f}\n\n"
            f"Achieved Singular Values:\n"
            f"  σ₀ = {sigma_achieved[0]:.4f}\n"
            f"  σ₁ = {sigma_achieved[1]:.4f}\n\n"
            f"Calibration Status:\n"
            f"  V† locked: {state.v_locked}\n"
            f"  U locked: {state.u_locked}\n"
            f"  Overall: {'LOCKED' if state.locked else 'NOT LOCKED'}\n\n"
            f"Total iterations: {state.iteration}\n"
            f"Final error: {state.error:.2e}"
        )
        ax9.text(0.1, 0.9, info_text, transform=ax9.transAxes,
                 fontsize=10, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        # Add overall title
        status = "LOCKED" if state.locked else "NOT LOCKED"
        plt.suptitle(
            f"{title}\n{status} after {state.iteration} iterations | "
            f"Final error: {state.error:.2e}",
            fontsize=14,
        )

        plt.tight_layout()
        return fig


def demo_visualizer():
    """Demo the visualizer with a complete calibration run."""
    from .control_loop import CoherentController, ControlConfig
    from plant import PhotonicPlant

    print("Running visualization demo...")

    # Create plant
    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=2.0,
        tau_thermal=1e-9,
        seed=42,
    )

    # Create controller
    config = ControlConfig(
        initial_step=0.5,
        min_step=0.01,
        error_threshold=1e-3,
        max_iterations=150,
    )
    controller = CoherentController(plant, config)

    # Set target (Hadamard-like)
    h = 1 / np.sqrt(2)
    controller.set_target(h, h, h, -h)

    # Run calibration
    print("Running calibration...")
    locked = controller.run_calibration()
    print(f"Calibration {'LOCKED' if locked else 'did not lock'}")

    # Create visualizer
    viz = ControlVisualizer()

    # Generate plots
    print("Generating plots...")

    fig1 = viz.plot_convergence(
        controller.state.error_history,
        threshold=config.error_threshold,
    )
    fig1.savefig("demo_convergence.png", dpi=150)
    print("  Saved: demo_convergence.png")

    fig2 = viz.plot_phase_trajectories(controller.state.phase_history)
    fig2.savefig("demo_phases.png", dpi=150)
    print("  Saved: demo_phases.png")

    fig3 = viz.plot_matrix_evolution(
        controller.state.matrix_history,
        controller.target_matrix,
    )
    fig3.savefig("demo_matrix_evolution.png", dpi=150)
    print("  Saved: demo_matrix_evolution.png")

    fig4 = viz.plot_iq_constellation(
        controller.state.matrix_history,
        controller.target_matrix,
    )
    fig4.savefig("demo_iq_constellation.png", dpi=150)
    print("  Saved: demo_iq_constellation.png")

    fig5 = viz.plot_summary(controller, "Hadamard-like Matrix Calibration")
    fig5.savefig("demo_summary.png", dpi=150)
    print("  Saved: demo_summary.png")

    plt.show()


if __name__ == "__main__":
    demo_visualizer()
