# Вариант 1, k=4: x[n+1] = r*x*(1 - (x + 4x^2)/5). Запуск: python model_variant1_k4.py
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve

f  = lambda x, r: r*x*(1 - (x + 4*x*x)/5)          # отображение
df = lambda x, r: r/5*(5 - 2*x - 12*x*x)           # производная
xp = lambda r: (-1 + np.sqrt(81 - 80/r))/8         # x*_+ (r>1)
x_c = (np.sqrt(649) - 3)/32                        # корень 16x^2+3x-10=0
r_c = 5/(5 - x_c - 4*x_c**2)                       # r_crit
c   = (-1 + np.sqrt(61))/12                        # точка максимума f

def cycle2(r):
    """Цикл периода 2: s=x1+x2 из кубического уравнения, затем x1,x2."""
    for s in np.roots([16*r, 8*r, -(19*r + 40), -5*(r + 1)]):
        if abs(s.imag) < 1e-12 and s.real > 0:
            s = s.real; p = (r*(4*s*s + s - 5) - 5)/(4*r); D = s*s - 4*p
            if D > 0: return np.array([(s - D**.5)/2, (s + D**.5)/2])
    return np.array([np.nan, np.nan])

mu = lambda r: df(*cycle2(r)[:1], r)*df(*cycle2(r)[1:], r)   # мультипликатор
rs = np.arange(r_c + 1e-5, 2.8, 5e-4)
br = np.array([[r, *cycle2(r), mu(r)] for r in rs])
r2 = brentq(lambda r: mu(r) + 1, 2.3, 2.5, xtol=1e-13)       # |mu|=1, 2-я бифуркация

def fN(x, r, N):
    for _ in range(N): x = f(x, r)
    return x
def dN(x, r, N):
    d = 1.0
    for _ in range(N): d *= df(x, r); x = f(x, r)
    return d
def attr(r, burn=20000):
    x = 0.2
    for _ in range(burn): x = f(x, r)
    return x
def r_next(N, rg):                                    # fN(x)=x, (fN)'(x)=-1
    x0 = attr(rg - 0.004)
    return fsolve(lambda v: [fN(v[0], v[1], N) - v[0], dN(v[0], v[1], N) + 1],
                  [x0, rg], xtol=1e-13)[1]
r3, r4 = r_next(4, 2.471), r_next(8, 2.4834)
delta1, delta2 = (r2 - r_c)/(r3 - r2), (r3 - r2)/(r4 - r3)

print(f"x_crit={x_c:.5f} r_crit={r_c:.5f} r2={r2:.5f} r3={r3:.5f} r4={r4:.5f}")
print(f"delta1={delta1:.3f} delta2={delta2:.3f}")
for m in (1.05, 1.1, 1.2):
    r = m*r_c; x1, x2 = cycle2(r)
    eps = max(abs(f(x1, r) - x2), abs(f(x2, r) - x1))
    print(f"{m}*rc={r:.4f} x1={x1:.5f} x2={x2:.5f} eps={eps:.1e} mu={mu(r):.4f}")
r = 2.7; x = 0.2
for _ in range(1000): x = f(x, r)
tr = np.empty(10000)
for i in range(10000): x = f(x, r); tr[i] = x
x = 0.2
for _ in range(1000): x = f(x, r)
s = 0.0
for _ in range(200000): x = f(x, r); s += np.log(abs(df(x, r)))
a, b = 0.2, 0.2 + 1e-10; n = 0
while abs(a - b) < 0.1: a, b, n = f(a, r), f(b, r), n + 1
f1 = f(c, r); f2 = f(f1, r); f3 = f(f2, r)
print(f"lyapunov={s/200000:.4f} diverge_after={n} f(c)={f1:.4f} f2(c)={f2:.4f} f3(c)={f3:.4f} r_max={5/(5*c-c*c-4*c**3):.4f}")
