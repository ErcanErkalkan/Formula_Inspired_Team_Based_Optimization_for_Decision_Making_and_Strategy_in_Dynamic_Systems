import matplotlib.pyplot as plt
import numpy as np

# Conceptual 2D decision space (x1, x2) for t=1, t=2, t=3
np.random.seed(42)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Define optimal regions (e.g., PS lines or points)
opt_t1 = np.array([0.2, 0.2])
opt_t2 = np.array([0.5, 0.5])
opt_t3 = np.array([0.8, 0.4])

# Define leaders
leaders_t1 = opt_t1 + np.random.randn(6, 2) * 0.05
leaders_t2 = opt_t2 + np.random.randn(6, 2) * 0.05
leaders_t3 = opt_t3 + np.random.randn(6, 2) * 0.05

# Define support individuals (scattered)
support_t1 = opt_t1 + np.random.randn(20, 2) * 0.15

# At t=2, before redeployment, support is at t1 position roughly
support_t2_old = support_t1 + np.random.randn(20, 2) * 0.05

# Anchors for t=2 (extrapolation from t1 to t2... wait, t1 to t2 isn't available until t2 leaders are found. At t2, we have pre-t2 leaders and pre-t1 leaders... wait. The drift is from t-1 to t. So at t=3, drift is L_t2 to L_t3)
# Let's show t=1 (converged), t=2 (change happens, leaders drift, support is left behind), t=2 (redeployment)
# Wait, let's do:
# Panel 1: t=1 (Converged). Leaders and Support around Opt_1
# Panel 2: t=2 (Change occurs). New Opt_2. Leaders have moved to Opt_2. Support is still stuck near Opt_1 (weak). Anchor is projected from L_1 to L_2.
# Panel 3: t=2 (Redeployment). Weak support individuals are moved to anchors and current leaders.

axes[0].set_title("Environment $t-1$ (Converged)", fontsize=14)
axes[0].scatter(opt_t1[0], opt_t1[1], color='black', marker='*', s=200, label='True PS')
axes[0].scatter(leaders_t1[:, 0], leaders_t1[:, 1], color='blue', s=60, label='Leaders $\mathcal{L}_{t-1}$')
axes[0].scatter(support_t1[:, 0], support_t1[:, 1], color='gray', alpha=0.6, s=30, label='Support')
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].legend(loc='lower right')

# Panel 2: Change to t. Leaders have tracked the change, support is trailing.
axes[1].set_title("Environment $t$ (After Change, Pre-Redeployment)", fontsize=14)
axes[1].scatter(opt_t2[0], opt_t2[1], color='black', marker='*', s=200)
axes[1].scatter(leaders_t2[:, 0], leaders_t2[:, 1], color='blue', s=60, label='Leaders $\mathcal{L}_{t}$')
axes[1].scatter(support_t1[:, 0], support_t1[:, 1], color='gray', alpha=0.6, s=30, label='Lagging Support')

# Drift vector
drift = np.mean(leaders_t2, axis=0) - np.mean(leaders_t1, axis=0)
axes[1].annotate("", xy=np.mean(leaders_t2, axis=0), xytext=np.mean(leaders_t1, axis=0),
                 arrowprops=dict(arrowstyle="->", color="black", ls="dashed"))
axes[1].text(0.35, 0.35, "Leader Drift", rotation=45, fontsize=10)

# Anchor
anchor = np.mean(leaders_t2, axis=0) + drift
axes[1].scatter(anchor[0], anchor[1], color='green', marker='X', s=150, label='Predictive Anchor $\mathcal{A}_t$')

axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].legend(loc='lower right')

# Panel 3: Redeployment
axes[2].set_title("Environment $t$ (Active Redeployment)", fontsize=14)
axes[2].scatter(opt_t2[0], opt_t2[1], color='black', marker='*', s=200)
axes[2].scatter(leaders_t2[:, 0], leaders_t2[:, 1], color='blue', s=60)
axes[2].scatter(anchor[0], anchor[1], color='green', marker='X', s=150)

# Good support stays
good_support = support_t1[:10] # some support
axes[2].scatter(good_support[:, 0], good_support[:, 1], color='gray', alpha=0.6, s=30)

# Redeployed support around anchor and leaders
redeploy_anchor = anchor + np.random.randn(5, 2) * 0.05
redeploy_leaders = np.mean(leaders_t2, axis=0) + np.random.randn(5, 2) * 0.05

axes[2].scatter(redeploy_anchor[:, 0], redeploy_anchor[:, 1], color='red', marker='^', s=60, label='Redeployed (Anchor)')
axes[2].scatter(redeploy_leaders[:, 0], redeploy_leaders[:, 1], color='orange', marker='^', s=60, label='Redeployed (Leader)')

axes[2].set_xlim(0, 1)
axes[2].set_ylim(0, 1)
axes[2].legend(loc='lower right')

for ax in axes:
    ax.set_xlabel("Decision Variable $x_1$")
    ax.set_ylabel("Decision Variable $x_2$")
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("experiments/results/redeployment_concept.pdf", format='pdf')
plt.close()
print("Plot saved to experiments/results/redeployment_concept.pdf")
