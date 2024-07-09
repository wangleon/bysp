import numpy as np
import matplotlib.pyplot as plt

def plot_image_with_hist(data, dpi=120, figsize=(10, 6), scale=(5, 95),
                         title='', show=True, figfilename=None):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    b1 = 0.10
    h1 = 0.8
    w1 = h1/figsize[0]*figsize[1]
    l2 = 0.66
    w2 = 0.30
    hgap1 = 0.08
    h3 = 0.02
    h2 = (h1-2*hgap1-h3)/2
    ax_image = fig.add_axes([b1, b1,  w1, h1])
    ax_hist1 = fig.add_axes([l2, b1+h3+hgap1*2+h2, w2, h2])
    ax_hist2 = fig.add_axes([l2, b1+h3+hgap1, w2, h2])
    ax_cbar0 = fig.add_axes([l2, b1,  w2, h3])

    vmin = np.percentile(data, scale[0])
    vmax = np.percentile(data, scale[1])
    cax = ax_image.imshow(data, origin='lower', vmin=vmin, vmax=vmax)
    fig.colorbar(cax, cax=ax_cbar0, orientation='horizontal')
    ax_image.set_xlabel('X (pixel)')
    ax_image.set_ylabel('Y (pixel)')

    # plot hist1, the whole histogram
    ax_hist1.hist(data.flatten(), bins=50)
    ax_hist1.axvline(x=vmin, color='k', ls='--', lw=0.7)
    ax_hist1.axvline(x=vmax, color='k', ls='--', lw=0.7)
    y1, y2 = ax_hist1.get_ylim()
    ax_hist1.text(vmin, 0.1*y1+0.9*y2, str(scale[0])+'%')
    ax_hist1.text(vmax, 0.1*y1+0.9*y2, str(scale[1])+'%')
    ax_hist1.set_ylim(y1, y2)
    # ax_hist1.set_yticklabels([])
    # ax_hist2.set_yticklabels([])
    ax_hist2.hist(data.flatten(), bins=np.linspace(vmin, vmax, 50))
    ax_hist2.set_xlim(vmin, vmax)
    fig.suptitle(title)

    if figfilename is not None:
        fig.savefig(figfilename)

    # if show:
    #     plt.show()

    plt.close(fig)
