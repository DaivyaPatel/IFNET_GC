import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import io

flat_real = np.random.randint(0, 255, 10000)
flat_interp = flat_real + np.random.randint(-10, 10, 10000)

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
fig.patch.set_facecolor('#050505')
ax.set_facecolor('#050505')

hb = ax.hexbin(flat_real, flat_interp, gridsize=50, cmap='inferno', mincnt=1)
ax.plot([0, 255], [0, 255], color='white', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal (y=x)')

ax.set_title("Real vs Interpolated Intensity", color='#e0e0e0', pad=15, fontsize=14, fontweight='bold')
ax.set_xlabel("Ground Truth Intensity", color='#a0a0a0', fontsize=12, labelpad=10)
ax.set_ylabel("Interpolated Intensity", color='#a0a0a0', fontsize=12, labelpad=10)
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.tick_params(colors='#a0a0a0', labelsize=10)
ax.grid(color='#ffffff', alpha=0.1, linestyle='--')

cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('Pixel Density', color='#a0a0a0', fontsize=12, labelpad=10)
cb.ax.yaxis.set_tick_params(color='#a0a0a0', labelsize=10)
plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#a0a0a0')

ax.legend(loc='upper left', frameon=False, labelcolor='#a0a0a0', fontsize=10)

ax.set_box_aspect(1)
fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)

plt.savefig("test_out.png", format='png', facecolor=fig.get_facecolor(), transparent=False, dpi=150)
print("Saved test_out.png")
