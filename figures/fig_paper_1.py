# The purpose of this script is to generate summary figures for perfect learning

# Trial using test_case_7

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



# FIG:  template for aliasing and undersampling 

gs = gridspec.GridSpec(2,11, left=0.06, right=0.97, top=0.95, hspace=0.4, 
                       wspace=1.6, bottom=0.1)

fig_var = plt.figure(figsize=(18,6))
ax_main = fig_var.add_subplot(gs[0:, 0:5])
subax = fig_var.add_axes([0.22, 0.19,0.2,0.4])

ax_loss1 = fig_var.add_subplot(gs[0, 5:7])
ax_kamp1 = fig_var.add_subplot(gs[0, 7:9])
ax_pred1 = fig_var.add_subplot(gs[0, 9:11])
ax_loss2 = fig_var.add_subplot(gs[1, 5:7])
ax_kamp2 = fig_var.add_subplot(gs[1, 7:9])
ax_pred2 = fig_var.add_subplot(gs[1, 9:11])





for ax in [ax_main, ax_loss1, ax_loss2, ax_kamp1, ax_kamp2, ax_pred1, ax_pred2]:
    ax.set(title="hello",xlabel="x", ylabel="y", ylim=[-1,1])
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
plt.show()
