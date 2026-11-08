import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":12,"axes.grid":True,"grid.alpha":0.3,"mathtext.fontset":"dejavusans"})
K=4
f=lambda x,r: r*x*(1-(x+K*x*x)/(1+K))
df=lambda x,r: r/5*(5-2*x-12*x*x)
xs=lambda r:(-1+np.sqrt(81-80/r))/8
xc=(np.sqrt(649)-3)/32; rc=5/(5-xc-4*xc**2)
from model_variant1_k4 import br, r2   # ветвь цикла и r2 из расчётного скрипта
def it(r,x0=0.2,n=90):
    v=np.empty(n+1);v[0]=x0
    for i in range(n): v[i+1]=f(v[i],r)
    return v
# fig1
fig,ax=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
for a,(m,lab) in zip(ax,[(0.9,"устойчивое равновесие"),(1.1,"цикл периода 2")]):
    r=m*rc; v=it(r); a.plot(v,"o-",ms=3,lw=1,label=r"$x_n$")
    if m<1: a.axhline(xs(r),ls="--",c="r",label=rf"$x^*_+={xs(r):.4f}$")
    else:
        c=br[np.argmin(abs(br[:,0]-r))]
        a.axhline(c[1],ls="--",c="r",label=rf"$x_1={c[1]:.4f}$");a.axhline(c[2],ls="--",c="g",label=rf"$x_2={c[2]:.4f}$")
    a.set_title(rf"$r={m}\,r_{{crit}}\approx{r:.3f}$: {lab}");a.set_xlabel("номер итерации $n$");a.set_ylabel("$x_n$");a.legend(loc="lower right")
fig.savefig("fig1.png",dpi=200);plt.close(fig)
# fig2 bifurcation
rv=np.linspace(1.0,2.8,3000); R=[];X=[]
x=np.full_like(rv,0.2)
for _ in range(1500): x=f(x,rv)
for _ in range(250):
    x=f(x,rv);R.append(rv.copy());X.append(x.copy())
R=np.concatenate(R);X=np.concatenate(X)
fig,ax=plt.subplots(figsize=(7,4.8),constrained_layout=True)
ax.plot(R,X,",",c="#1f4e9c",alpha=0.5)
ax.axvline(rc,ls="--",c="r",label=rf"$r_{{crit}}\approx{rc:.3f}$ (1-я бифуркация)")
ax.axvline(r2,ls="--",c="g",label=rf"$r_2\approx{r2:.3f}$ (2-я бифуркация)")
ax.axvline(2.66,ls=":",c="k",label=r"окно периода 3 ($r\approx2.66$)")
ax.set_xlim(1,2.8);ax.set_ylim(0,1.02);ax.set_xlabel("параметр $r$");ax.set_ylabel("установившиеся значения $x_n$")
ax.set_title("Бифуркационная диаграмма, $k=4$ (отброшено 1500 итераций)");ax.legend(loc="upper left")
fig.savefig("fig2.png",dpi=220);plt.close(fig)
# fig3 cycle branches
fig,ax=plt.subplots(figsize=(7,4.8),constrained_layout=True)
m=(br[:,0]<=2.7); b=br[m]; st=abs(b[:,3])<1
ra=np.linspace(1,rc,200); ax.plot(ra,xs(ra),"k-",label=r"$x^*_+(r)$ устойчива")
rb=np.linspace(rc,2.7,300); ax.plot(rb,xs(rb),"k--",label=r"$x^*_+(r)$ неустойчива")
for j,c in [(1,"C0"),(2,"C1")]:
    ax.plot(b[st,0],b[st,j],c=c,lw=2,label=rf"$x_{j}(r)$ устойчив")
    ax.plot(b[~st,0],b[~st,j],c=c,ls="--",lw=1.2,label=rf"$x_{j}(r)$ неустойчив")
ax.axvline(rc,ls=":",c="r");ax.axvline(r2,ls=":",c="g")
ax.text(rc+0.01,0.95,r"$r_{crit}$",color="r");ax.text(r2+0.01,0.95,r"$r_2$",color="g")
ax.set_xlim(1.9,2.7);ax.set_ylim(0.1,1.0);ax.set_xlabel("параметр $r$");ax.set_ylabel("$x$")
ax.set_title("Ветви цикла периода 2 и неподвижной точки");ax.legend(loc="lower left",ncol=2,fontsize=9)
fig.savefig("fig3.png",dpi=220);plt.close(fig)
# chaos
r=2.7; x=0.2
for _ in range(1000): x=f(x,r)
tr=np.empty(10000)
for i in range(10000): x=f(x,r);tr[i]=x
fig,ax=plt.subplots(figsize=(10.5,4),constrained_layout=True)
ax.plot(np.arange(1000,1300),tr[:300],lw=0.8);ax.set_xlabel("номер итерации $n$");ax.set_ylabel("$x_n$")
ax.set_title("Временной ряд при $r=2.7$ (после отброса 1000 итераций)");ax.set_ylim(0,1.02)
fig.savefig("fig4.png",dpi=200);plt.close(fig)
c=(-1+np.sqrt(61))/12;f1=f(c,r);f2=f(f1,r);f3=f(f2,r)
fig,ax=plt.subplots(figsize=(10.5,4.2),constrained_layout=True)
ax.hist(tr,bins=80,density=True,color="#1f77b4",edgecolor="k",lw=0.25)
for v,t in [(f2,r"$f^2(c)$"),(f3,r"$f^3(c)$"),(f1,r"$f(c)$")]:
    ax.axvline(v,ls="--",c="r",lw=1);ax.text(v+0.005,ax.get_ylim()[1]*0.93,t,color="r")
ax.axvspan(0,f2,color="gray",alpha=0.25,label="запрещённые зоны");ax.axvspan(f1,1.0,color="gray",alpha=0.25)
ax.set_xlim(0,1);ax.set_xlabel("значение $x_n$");ax.set_ylabel("плотность");ax.legend(loc="upper center")
ax.set_title("Гистограмма значений при $r=2.7$ (10 000 итераций, 80 бинов)")
fig.savefig("fig5.png",dpi=200);plt.close(fig)
print(f1,f2,f3)
