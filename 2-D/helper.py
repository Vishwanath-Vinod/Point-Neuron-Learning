import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.ndimage import gaussian_filter

# Adding AWGN to Signal
def awgn(signal, snr_db):
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise

# Plotting
def plot_field(PLACAR, title, value, filename, clim=None, cmap=None, cbar_ticks=None):

    # Mesh Plot
    SizeX, SizeY = 1.3, 1.3
    Resolution = 0.02
    x = np.arange(-SizeX, SizeX + Resolution, Resolution)
    y = np.arange(-SizeY, SizeY + Resolution, Resolution)
    Px, Py = np.meshgrid(x, y)
    plt.figure()

    #Heatmap of given field
    field = griddata(PLACAR, value, (Px, Py), method='cubic')
    plt.pcolormesh(Px, Py, np.real(field), shading='auto', cmap=cmap)
    cbar = plt.colorbar()
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.xticks(np.linspace(-1.2, 1.2, 5))
    plt.yticks(np.linspace(-1.2, 1.2, 5))

    # Target region boundary
    theta = np.linspace(0, 2 * np.pi, 1000)
    targetX = np.cos(theta)
    targetY = np.sin(theta)
    plt.plot(targetX, targetY, 'k-', linewidth=1)

    plt.title(title)
    if clim:
        plt.clim(clim)
    plt.savefig(filename, dpi=300, bbox_inches='tight')

def plot_loss(Loss,results_dir):
    plt.figure(figsize=(8, 5))
    ax_main = plt.gca()
    ax_main.plot(Loss, label='Loss')
    ax_main.set_title("Loss Curve with Zoom-In View")
    ax_main.set_xlabel("Iteration number")
    ax_main.set_ylabel("Loss")
    ax_main.grid(False)

    ax_inset = inset_axes(ax_main, width="50%", height="50%", loc="upper right", borderpad=2)
    ax_inset.plot(Loss, color='tab:blue')
    ax_inset.set_xlim(200, 3000)
    ax_inset.set_ylim(0.012, 0.03)
    ax_inset.set_title("Zoomed-In", fontsize=8)
    ax_inset.grid(True)
    mark_inset(ax_main, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.tight_layout()
    plt.savefig(f'{results_dir}/zoomed_loss.png')

def plot_compare():
    loss_manual = np.loadtxt('Results/Manual/loss.csv')
    loss_auto = np.loadtxt('Results/Auto/loss.csv')
    with open('Results/Manual/time.txt') as f:
        time_manual = float(f.read())
    with open('Results/Auto/time.txt') as f:
        time_auto = float(f.read())

    plt.figure(figsize=(8, 5))

    # Plot both losses
    plt.plot(loss_manual, label='Manual Gradient Computation', color='blue')
    plt.plot(loss_auto, label='Autograd', color='red')

    # Log scale on y-axis
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Convergence Comparison (Log Scale)")
    plt.grid(True)
    plt.tight_layout()

    # Add final value annotations
    manual_line = f"Manual        {loss_manual[-1]:>10.2e}     {time_manual:>6.1f}s"
    auto_line   = f"Auto          {loss_auto[-1]:>10.2e}     {time_auto:>6.1f}s"
    header      = "Method        Final Loss       Time"
    full_text   = f"{header}\n{manual_line}\n{auto_line}"

    # Draw white box
    plt.gca().text(0.02, 0.98, full_text,
                   transform=plt.gca().transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                   family='monospace')  # monospaced font for alignment

    plt.tight_layout()
    plt.legend()
    plt.savefig('Results/Compare/loss_comparison.png')

if __name__ == "__main__":
    plot_compare()
