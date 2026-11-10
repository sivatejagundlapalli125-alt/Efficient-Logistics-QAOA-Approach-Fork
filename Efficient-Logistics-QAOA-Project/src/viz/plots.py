import numpy as np
import matplotlib.pyplot as plt

def plot_cost_heatmap(matrix,names,output_path=None):
    m=np.asarray(matrix); fig,ax=plt.subplots(figsize=(7,5)); im=ax.imshow(m,cmap="YlGnBu")
    ax.set_xticks(range(len(names)),names,rotation=35,ha="right"); ax.set_yticks(range(len(names)),names)
    ax.set_title("Travel cost between locations")
    for i in range(len(names)):
        for j in range(len(names)): ax.text(j,i,f"{m[i,j]:.0f}",ha="center",va="center",fontsize=8)
    fig.colorbar(im,ax=ax,label="Cost"); fig.tight_layout()
    if output_path: fig.savefig(output_path,dpi=150,bbox_inches="tight")
    return fig

def plot_route_graph(route,names,matrix,output_path=None):
    n=len(names); ang=np.linspace(0,2*np.pi,n,endpoint=False); pts=np.c_[np.cos(ang),np.sin(ang)]
    fig,ax=plt.subplots(figsize=(7,5))
    for i,name in enumerate(names):
        ax.scatter(*pts[i],s=700,color="#ccebd8",edgecolor="#16834a",zorder=3); ax.text(*pts[i],name,ha="center",va="center",fontsize=8,zorder=4)
    cyc=list(route)+[route[0]]
    for a,b in zip(cyc,cyc[1:]): ax.annotate("",xy=pts[b],xytext=pts[a],arrowprops={"arrowstyle":"->","color":"#176b87","lw":2,"shrinkA":24,"shrinkB":24})
    ax.set_title("Optimized delivery cycle"); ax.axis("off"); ax.set_aspect("equal"); fig.tight_layout()
    if output_path: fig.savefig(output_path,dpi=150,bbox_inches="tight")
    return fig
