import matplotlib.lines as mlines
import matplotlib.patches as mpatches

########################################
# FIG: Size, Colors and Labels
########################################
fsize=13.5
PLOT_SCALE = 1000
optimal_star = 'magenta'
savefig='Yes'
us_colour_list = ['dodgerblue', 'blue', 'purple', 'olive', 'darkorange']

style = ['-', '-', '-', '-']
ax_kea_labels=['A', 'B', 'C', 'D', 'E']
ax_tui_labels=['A*', 'B*', 'C*', 'D*', 'E*']

loss_color_list = ['tan', 'c', optimal_star, optimal_star]

kf_label = 'KF - Basis of Oscillators'
akf_label = 'KF - Autoregressive AR(q=101) with LS Weights'
ls_label = 'LS (past msmts: q=101)'
akf_color = 'k'
ls_color = 'g'

true_band_edge_color = 'r'
kf_basis_edge_color = 'teal'
########################################
# FIG: Custom Legends
########################################
l_train = mpatches.Patch(color='gray', alpha=0.3)
predictzeroline =  mlines.Line2D([], [], linestyle='-', color='darkblue')

randinit = mlines.Line2D([], [], linestyle='None', color=None, marker='v', markerfacecolor='k', markeredgecolor='k', markersize=7)

optimalstar = mlines.Line2D([], [], linestyle='None',  color=None, marker='*', markerfacecolor=optimal_star, markeredgecolor=optimal_star, markersize=7, alpha=1)

pred_circ = mlines.Line2D([], [], linestyle='-',  color='tan', marker='o', markerfacecolor='tan', markeredgecolor='tan', markersize=7)
pred_line = mlines.Line2D([], [], linestyle='-',  color=optimal_star)

fore_circ = mlines.Line2D([], [], linestyle='-',  color='c', marker='o', markerfacecolor='c', markeredgecolor='c', markersize=7)
fore_line = mlines.Line2D([], [], linestyle='-',  color='teal')

un_opt_traj = mlines.Line2D([], [], linestyle='-', color='gray')
opt_traj = mlines.Line2D([], [], linestyle='-', color=optimal_star)
