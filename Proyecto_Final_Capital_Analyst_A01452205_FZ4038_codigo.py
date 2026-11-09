# -*- coding: utf-8 -*-
"""
FZ4038 – Proyecto Final – Modelos – Capital Analyst
Código del notebook Proyecto_Final_Capital_Analyst_A01452205_FZ4038.ipynb (solo código, sin resultados).
Alumno: Daniel Mondragón Monroy (A01452205) · Econometría Financiera, EGADE Business School
Los resultados, gráficas e interpretaciones están en el archivo .ipynb y en el HTML de este repositorio.
"""


# ======================================================================
# Proyecto Final – Modelos – Capital Analyst
# ======================================================================

# ======================================================================
# Automatización de algoritmos para el portafolio de Capital Analyst: regresión, datos panel, ARIMA, GARCH y simulación
# ======================================================================

# ======================================================================
# 1. El caso Capital Analyst y objetivo
# ======================================================================
# Paquetes (Colab ya trae numpy, pandas, statsmodels, scipy y matplotlib)
# !pip -q install yfinance arch linearmodels

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from arch import arch_model
from linearmodels.panel import PooledOLS, PanelOLS, RandomEffects
from IPython.display import display
warnings.simplefilter("ignore")              # después de importar: statsmodels redefine sus avisos al cargarse

plt.rcParams["figure.figsize"] = (11, 4)
plt.rcParams["axes.grid"] = True
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 30)

def estrellas(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ======================================================================
# 2. Universo de inversión y datos
# ======================================================================

# ======================================================================
# 2.1 Universo de inversión
# ======================================================================

# ======================================================================
# 2.2 Fuentes y variables
# ======================================================================
import yfinance as yf

ACTIVOS = ["AAPL", "MSFT", "NVDA", "GOOGL", "JPM", "GS", "BAC", "XOM", "CVX", "COP", "USO"]
SECTOR = {"AAPL": "Tecnología", "MSFT": "Tecnología", "NVDA": "Tecnología", "GOOGL": "Tecnología",
          "JPM": "Bancos", "GS": "Bancos", "BAC": "Bancos",
          "XOM": "Energía", "CVX": "Energía", "COP": "Energía", "USO": "ETF petróleo"}
FACTORES = {"^GSPC": "mercado", "^TNX": "tasa10", "BZ=F": "brent", "DX-Y.NYB": "dolar", "^IRX": "tbill"}
INICIO, FIN = "2015-01-01", "2026-11-07"           # FIN no se incluye: último dato 6-nov-2026

todo = yf.download(ACTIVOS + list(FACTORES), start=INICIO, end=FIN, auto_adjust=True, progress=False)["Close"]
faltan = [t for t in ACTIVOS + list(FACTORES) if t not in todo.columns or todo[t].dropna().empty]
if faltan:
    raise RuntimeError(f"No se descargaron: {faltan}. Vuelve a correr la celda.")

# Limpieza: los activos y los factores cotizan en calendarios ligeramente distintos (futuros, índices)
precios = todo[ACTIVOS].dropna()
fact_niv = todo[list(FACTORES)].rename(columns=FACTORES).ffill().reindex(precios.index)
if fact_niv["tasa10"].median() > 20:                 # algunas versiones reportan ^TNX multiplicado por 10
    fact_niv["tasa10"] /= 10; fact_niv["tbill"] /= 10

r = 100 * np.log(precios).diff().dropna()            # rendimientos log diarios en %
fact = pd.DataFrame({
    "mercado": 100 * np.log(fact_niv["mercado"]).diff(),
    "tasa10": fact_niv["tasa10"].diff(),              # cambio en puntos porcentuales
    "brent": 100 * np.log(fact_niv["brent"]).diff(),
    "dolar": 100 * np.log(fact_niv["dolar"]).diff(),
}).reindex(r.index)
rf_anual = fact_niv["tbill"].mean()                  # % anual promedio del periodo

print(f"Periodo: {precios.index[0].date()} a {precios.index[-1].date()} | {len(precios):,} días hábiles")
print("Faltantes eliminados en activos:", int(todo[ACTIVOS].isna().any(axis=1).sum()), "días")
print(f"Tasa libre de riesgo promedio (T-Bill 3 meses): {rf_anual:.2f}% anual")
precios.to_csv("precios_diarios_2015_2026.csv")


# ======================================================================
# 2.3 Calidad de los datos
# ======================================================================
calidad = pd.DataFrame({
    "Observaciones": precios.count(),
    "Precios ≤ 0": (precios <= 0).sum(),
    "Rend. mínimo (%)": r.min().round(2),
    "Fecha mínimo": r.idxmin().dt.date,
    "Rend. máximo (%)": r.max().round(2),
    "Fecha máximo": r.idxmax().dt.date,
})
display(calidad)


# ======================================================================
# 3. Métricas de riesgo y rendimiento
# ======================================================================
def max_caida(serie_precio):
    return (serie_precio / serie_precio.cummax() - 1).min() * 100

riesgo = pd.DataFrame({
    "Sector": pd.Series(SECTOR),
    "Rend. anual (%)": r.mean() * 252,
    "Volatilidad anual (%)": r.std() * np.sqrt(252),
    "Sharpe": (r.mean() * 252 - rf_anual) / (r.std() * np.sqrt(252)),
    "Máx. caída (%)": precios.apply(max_caida),
    "VaR 95% 1 día (%)": -r.quantile(0.05),
    "CVaR 95% 1 día (%)": -r[r <= r.quantile(0.05)].mean(),
}).loc[ACTIVOS]
display(riesgo.round(2))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
colores = {"Tecnología": "tab:blue", "Bancos": "tab:green", "Energía": "tab:orange", "ETF petróleo": "tab:red"}
for k in ACTIVOS:
    ax[0].scatter(riesgo.loc[k, "Volatilidad anual (%)"], riesgo.loc[k, "Rend. anual (%)"], s=60, color=colores[SECTOR[k]])
    ax[0].annotate(k, (riesgo.loc[k, "Volatilidad anual (%)"], riesgo.loc[k, "Rend. anual (%)"]), xytext=(4, 4), textcoords="offset points")
ax[0].set_xlabel("Volatilidad anual (%)"); ax[0].set_ylabel("Rendimiento anual (%)"); ax[0].set_title("Riesgo contra rendimiento")
corr = r.corr()
im = ax[1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax[1].set_xticks(range(len(ACTIVOS))); ax[1].set_xticklabels(ACTIVOS, rotation=90)
ax[1].set_yticks(range(len(ACTIVOS))); ax[1].set_yticklabels(ACTIVOS)
for i in range(len(ACTIVOS)):
    for j in range(len(ACTIVOS)):
        ax[1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax[1].set_title("Correlación de rendimientos diarios"); plt.colorbar(im, ax=ax[1], fraction=0.046)
plt.tight_layout(); plt.show()


# ======================================================================
# 4. Regresión lineal simple: modelo de mercado
# ======================================================================

# ======================================================================
# 4.1 Especificación
# ======================================================================
def pruebas(modelo, X):
    resid = modelo.resid
    return {"BP (p)": het_breuschpagan(resid, X)[1],
            "BG 5 rez. (p)": acorr_breusch_godfrey(modelo, nlags=5)[1],
            "JB (p)": jarque_bera(resid)[1],
            "DW": durbin_watson(resid)}

filas, mod_simple = [], {}
for k in ACTIVOS:
    datos = pd.concat([r[k], fact["mercado"]], axis=1).dropna()
    X = sm.add_constant(datos["mercado"])
    ols = sm.OLS(datos[k], X).fit()
    hac = sm.OLS(datos[k], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    mod_simple[k] = hac
    filas.append({"Activo": k, "Sector": SECTOR[k],
                  "α diario (%)": hac.params["const"], "p α": hac.pvalues["const"],
                  "β": hac.params["mercado"], "t β (HAC)": hac.tvalues["mercado"], "p β": hac.pvalues["mercado"],
                  "R²": ols.rsquared, "Riesgo sistemático (%)": 100 * ols.rsquared, **pruebas(ols, X)})
simple = pd.DataFrame(filas).set_index("Activo")
display(simple.round(4))

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
simple["β"].plot.bar(ax=ax[0], color=[colores[SECTOR[k]] for k in simple.index])
ax[0].axhline(1, color="k", ls="--", lw=1); ax[0].set_title("Beta de mercado por activo (línea: β = 1)")
k = "NVDA"
datos = pd.concat([r[k], fact["mercado"]], axis=1).dropna()
ax[1].scatter(datos["mercado"], datos[k], s=4, alpha=0.3)
xs = np.linspace(datos["mercado"].min(), datos["mercado"].max(), 50)
ax[1].plot(xs, mod_simple[k].params["const"] + mod_simple[k].params["mercado"] * xs, "r", lw=2)
ax[1].set_xlabel("r_m (S&P 500, %)"); ax[1].set_ylabel(f"r ({k}, %)"); ax[1].set_title(f"Modelo de mercado: {k}")
plt.tight_layout(); plt.show()


# ======================================================================
# 5. Regresión lineal múltiple: modelo de factores
# ======================================================================

# ======================================================================
# 5.1 Selección de variables y signos esperados
# ======================================================================

# ======================================================================
# 5.2 Multicolinealidad de los factores
# ======================================================================
F = fact[["mercado", "tasa10", "brent", "dolar"]].dropna()
Xf = sm.add_constant(F)
vif = pd.DataFrame({"VIF": [variance_inflation_factor(Xf.values, i) for i in range(1, Xf.shape[1])]}, index=F.columns)
display(F.corr().round(2)); display(vif.round(2))


# ======================================================================
# 5.3 Estimación por activo
# ======================================================================
filas, mod_mult = [], {}
for k in ACTIVOS:
    datos = pd.concat([r[k], F], axis=1).dropna()
    X = sm.add_constant(datos[F.columns])
    ols = sm.OLS(datos[k], X).fit()
    hac = sm.OLS(datos[k], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    mod_mult[k] = hac
    wald = hac.wald_test("tasa10 = 0, brent = 0, dolar = 0", scalar=True)
    fila = {"Activo": k, "Sector": SECTOR[k]}
    for v in F.columns:
        fila[v] = f"{hac.params[v]:.3f}{estrellas(hac.pvalues[v])}"
    fila.update({"R² simple": simple.loc[k, "R²"], "R² múltiple": ols.rsquared, "R² aj. múltiple": ols.rsquared_adj,
                 "p factores adicionales (Wald HAC)": float(wald.pvalue), **pruebas(ols, X)})
    filas.append(fila)
mult = pd.DataFrame(filas).set_index("Activo")
display(mult.round(4))
print("Significancia con errores HAC: *** 1%, ** 5%, * 10%. Coeficientes: tasa10 en % por punto porcentual; los demás son elasticidades diarias.")


# ======================================================================
# 6. Datos panel: ¿qué fundamentales explican el rendimiento de las acciones?
# ======================================================================

# ======================================================================
# 6.1 Planteamiento
# ======================================================================
# Datos trimestrales de LSEG (archivo del repositorio; si existe una copia local en Colab, se usa esa)
URL = "https://raw.githubusercontent.com/a01452205/FZ4038-Proyecto-Final/main/panel_trimestral_LSEG.csv"
import os
fuente = "panel_trimestral_LSEG.csv" if os.path.exists("panel_trimestral_LSEG.csv") else URL
lseg = pd.read_csv(fuente, parse_dates=["fecha_cierre"])
print("Fuente:", fuente, "|", lseg.shape)

lseg = lseg.sort_values(["ticker", "trimestre"])
lseg["crec"] = 100 * np.log(lseg["ingresos"] / lseg.groupby("ticker")["ingresos"].shift(4))
lseg["roe"] = 400 * lseg["utilidad_neta"] / lseg["capital_comun"]
lseg["apal"] = lseg["activo_total"] / lseg["capital_comun"]

# Rendimiento trimestral de la acción y del mercado en el trimestre siguiente
pq = np.log(pd.concat([precios, fact_niv["mercado"]], axis=1)).groupby(precios.index.to_period("Q")).last()
rq = 100 * pq.diff()
rq.index = rq.index.astype(str)
sig = rq.shift(-1)                                    # rendimiento del trimestre t+1
lseg["ret_sig"] = [sig.loc[q, t] if q in sig.index else np.nan for t, q in zip(lseg["ticker"], lseg["trimestre"])]
lseg["mkt_sig"] = [sig.loc[q, "mercado"] if q in sig.index else np.nan for q in lseg["trimestre"]]

panel = lseg.dropna(subset=["crec", "roe", "apal", "ret_sig", "mkt_sig"]).copy()
panel["periodo"] = pd.PeriodIndex(panel["trimestre"], freq="Q").to_timestamp()
panel = panel.set_index(["ticker", "periodo"])
print(f"Panel: {panel.index.get_level_values(0).nunique()} empresas, "
      f"{panel.index.get_level_values(1).nunique()} trimestres, {len(panel)} observaciones")
display(panel[["ret_sig", "crec", "roe", "apal", "mkt_sig"]].describe().T.round(2))


# ======================================================================
# 6.2 Modelos agrupado, de efectos fijos y de efectos aleatorios
# ======================================================================
y = panel["ret_sig"]
X = sm.add_constant(panel[["crec", "roe", "apal", "mkt_sig"]])
pooled = PooledOLS(y, X).fit()
fe = PanelOLS(y, X, entity_effects=True).fit()
re = RandomEffects(y, X).fit()

def tabla_panel(res, nombre):
    return pd.DataFrame({f"{nombre}: coef.": res.params, f"{nombre}: p": res.pvalues})
comp = pd.concat([tabla_panel(pooled, "Agrupado"), tabla_panel(fe, "Efectos fijos"), tabla_panel(re, "Efectos aleatorios")], axis=1)
display(comp.round(4))
print(f"R² (within) efectos fijos: {fe.rsquared_within:.3f} | R² agrupado: {pooled.rsquared:.3f}")


# ======================================================================
# 6.3 Elección del modelo: prueba F y prueba de Hausman
# ======================================================================
fp = fe.f_pooled
print(f"Prueba F de efectos individuales: F = {fp.stat:.3f}, p = {fp.pval:.4f}")

comunes = ["crec", "roe", "apal", "mkt_sig"]
b = fe.params[comunes] - re.params[comunes]
V = fe.cov.loc[comunes, comunes] - re.cov.loc[comunes, comunes]
h = float(b.T @ np.linalg.pinv(V) @ b)
p_h = 1 - stats.chi2.cdf(h, len(comunes))
print(f"Prueba de Hausman: χ² = {h:.3f}, gl = {len(comunes)}, p = {p_h:.4f}")


# ======================================================================
# 6.4 Pruebas de autocorrelación y dependencia transversal
# ======================================================================
# Wooldridge: regresión en primeras diferencias y prueba de que la autocorrelación de sus residuos es -0.5
dpanel = panel[["ret_sig", "crec", "roe", "apal", "mkt_sig"]].groupby(level=0).diff().dropna()
m_d = sm.OLS(dpanel["ret_sig"], dpanel[["crec", "roe", "apal", "mkt_sig"]]).fit()
e = m_d.resid
e_lag = e.groupby(level=0).shift(1)
dd = pd.concat([e.rename("e"), e_lag.rename("e1")], axis=1).dropna()
m_w = sm.OLS(dd["e"], dd["e1"]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.index.get_level_values(0))[0]})
w_stat = ((m_w.params["e1"] + 0.5) / m_w.bse["e1"]) ** 2
print(f"Wooldridge: coef. = {m_w.params['e1']:.3f} (bajo H0: -0.5), F = {w_stat:.3f}, p = {1 - stats.chi2.cdf(w_stat, 1):.4f}")

# Pesaran CD sobre los residuos de efectos fijos
res_fe = fe.resids.unstack(level=0)
N, T = res_fe.shape[1], res_fe.shape[0]
rho = res_fe.corr(min_periods=8).values
suma = np.nansum(np.triu(rho, 1))
cd = np.sqrt(2 * T / (N * (N - 1))) * suma
print(f"Pesaran CD = {cd:.3f}, p = {2 * (1 - stats.norm.cdf(abs(cd))):.4f} | correlación promedio entre empresas = {np.nanmean(rho[np.triu_indices(N, 1)]):.3f}")


# ======================================================================
# 6.5 Modelo final con errores estándar de Driscoll-Kraay
# ======================================================================
# Regla de decisión: efectos fijos si la prueba F o la de Hausman rechazan (p < 0.05); si no, el modelo agrupado.
# (Efectos fijos es consistente haya o no efectos individuales; con más de 40 trimestres por empresa su costo en eficiencia es pequeño.)
usar_fe = (fp.pval < 0.05) or (p_h < 0.05)
print("Modelo final:", "efectos fijos" if usar_fe else "agrupado", "con errores de Driscoll-Kraay")
dk = PanelOLS(y, X, entity_effects=usar_fe).fit(cov_type="kernel")
tabla_dk = pd.DataFrame({"Coeficiente": dk.params, "Error est. (DK)": dk.std_errors, "t": dk.tstats, "p": dk.pvalues})
tabla_dk["Sig."] = tabla_dk["p"].apply(estrellas)
display(tabla_dk.round(4))
print(f"R² = {dk.rsquared:.3f} | R² within = {dk.rsquared_within:.3f} | N = {dk.nobs}")


# ======================================================================
# 7. Series de tiempo: ARIMA y pronóstico
# ======================================================================

# ======================================================================
# 7.1 Ejemplo detallado: Microsoft
# ======================================================================
ACTIVO, H = "MSFT", 60
yv = np.log(precios[ACTIVO])
print(f"ADF log-precio p = {adfuller(yv, regression='ct', autolag='AIC')[1]:.4f} | ADF rendimiento p = {adfuller(yv.diff().dropna(), autolag='AIC')[1]:.4f}")
fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))
plot_acf(r[ACTIVO], lags=30, ax=ax[0], zero=False, title=f"ACF del rendimiento de {ACTIVO}")
plot_pacf(r[ACTIVO], lags=30, ax=ax[1], zero=False, method="ywm", title=f"PACF del rendimiento de {ACTIVO}")
for a in ax: a.set_ylim(-0.15, 0.15)
plt.tight_layout(); plt.show()

def seleccionar_arima(serie, pmax=3, qmax=3):
    res = []
    for p_ in range(pmax + 1):
        for q_ in range(qmax + 1):
            try:
                m = ARIMA(serie.values, order=(p_, 1, q_), trend="t").fit()
                res.append({"p": p_, "q": q_, "AIC": m.aic, "BIC": m.bic})
            except Exception:
                pass
    return pd.DataFrame(res).sort_values("BIC").reset_index(drop=True)

y_tr, y_te = yv.iloc[:-H], yv.iloc[-H:]
g = seleccionar_arima(y_tr)
display(g.head(5).round(1))
p_i, q_i = int(g.loc[0, "p"]), int(g.loc[0, "q"])
mod = ARIMA(y_tr.values, order=(p_i, 1, q_i), trend="t").fit()
print(f"ARIMA elegido por BIC: ({p_i},1,{q_i})"); print(mod.summary().tables[1])
lb = acorr_ljungbox(pd.Series(mod.resid[1:]), lags=[10, 20], model_df=p_i + q_i, return_df=True)
display(lb.round(4))

fc = mod.get_forecast(H)
media, ic = np.exp(fc.predicted_mean), np.exp(fc.conf_int(alpha=0.05))
real, ingenuo = np.exp(y_te.values), np.repeat(np.exp(y_tr.iloc[-1]), H)
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
print(f"RMSE ARIMA = {rmse(media, real):.3f} | RMSE ingenuo = {rmse(ingenuo, real):.3f} | cobertura IC 95% = {np.mean((real >= ic[:, 0]) & (real <= ic[:, 1])):.0%}")
plt.figure(figsize=(11, 4.5))
plt.plot(precios[ACTIVO].iloc[-250:], label="Precio observado")
plt.plot(y_te.index, media, "r--", label="Pronóstico ARIMA (fuera de muestra)")
plt.fill_between(y_te.index, ic[:, 0], ic[:, 1], color="r", alpha=0.15, label="IC 95%")
plt.title(f"{ACTIVO}: pronóstico ARIMA({p_i},1,{q_i}) de {H} días"); plt.legend(); plt.show()


# ======================================================================
# 7.2 ARIMA para todo el universo
# ======================================================================
filas, pron20 = [], {}
for k in ACTIVOS:
    yk = np.log(precios[k]); tr, te = yk.iloc[:-H], yk.iloc[-H:]
    gk = seleccionar_arima(tr)
    pk, qk = int(gk.loc[0, "p"]), int(gk.loc[0, "q"])
    mk = ARIMA(tr.values, order=(pk, 1, qk), trend="t").fit()
    f = mk.get_forecast(H); m_, c_ = np.exp(f.predicted_mean), np.exp(f.conf_int(alpha=0.05))
    re_, ing_ = np.exp(te.values), np.repeat(np.exp(tr.iloc[-1]), H)
    lbp = acorr_ljungbox(pd.Series(mk.resid[1:]), lags=[10], model_df=pk + qk, return_df=True)["lb_pvalue"].iloc[0]
    full = ARIMA(yk.values, order=(pk, 1, qk), trend="t").fit().get_forecast(20)
    pron20[k] = np.exp(full.predicted_mean)
    filas.append({"Activo": k, "ARIMA": f"({pk},1,{qk})", "Ljung-Box 10 (p)": lbp,
                  "RMSE ARIMA": rmse(m_, re_), "RMSE ingenuo": rmse(ing_, re_),
                  "ARIMA / ingenuo": rmse(m_, re_) / rmse(ing_, re_),
                  "Cobertura IC 95%": np.mean((re_ >= c_[:, 0]) & (re_ <= c_[:, 1])),
                  "Último precio": precios[k].iloc[-1], "Pronóstico 20 días": pron20[k][-1],
                  "Cambio esperado (%)": 100 * (pron20[k][-1] / precios[k].iloc[-1] - 1)})
arima_tab = pd.DataFrame(filas).set_index("Activo")
display(arima_tab.round(3))


# ======================================================================
# 8. Volatilidad: GARCH (puntos extra)
# ======================================================================
filas, garch = [], {}
for k in ACTIVOS:
    g11 = arch_model(r[k], mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    gjr = arch_model(r[k], mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    garch[k] = g11
    a_, b_ = g11.params["alpha[1]"], g11.params["beta[1]"]
    v20 = np.sqrt(g11.forecast(horizon=20).variance.iloc[-1].values.mean())
    filas.append({"Activo": k, "ARCH-LM (p)": het_arch(r[k] - r[k].mean(), nlags=10)[1],
                  "α": a_, "β": b_, "α+β": a_ + b_, "ν (colas)": g11.params["nu"],
                  "Vida media (días)": np.log(0.5) / np.log(a_ + b_) if a_ + b_ < 1 else np.nan,
                  "γ GJR": gjr.params["gamma[1]"], "p γ": gjr.pvalues["gamma[1]"],
                  "AIC GARCH": g11.aic, "AIC GJR": gjr.aic,
                  "Vol. actual anual (%)": g11.conditional_volatility.iloc[-1] * np.sqrt(252),
                  "Vol. pronosticada 20 días anual (%)": v20 * np.sqrt(252),
                  "Vol. histórica anual (%)": r[k].std() * np.sqrt(252)})
garch_tab = pd.DataFrame(filas).set_index("Activo")
display(garch_tab.round(3))

fig, ax = plt.subplots(figsize=(12, 4.5))
for k in ["NVDA", "JPM", "XOM", "USO"]:
    (garch[k].conditional_volatility * np.sqrt(252)).plot(ax=ax, lw=0.8, label=k)
ax.set_title("Volatilidad condicional anualizada (%) – GARCH(1,1)-t"); ax.legend(); plt.show()


# ======================================================================
# 9. Portafolio: covarianza pronosticada, comparación y simulación de escenarios
# ======================================================================

# ======================================================================
# 9.1 Portafolios comparados
# ======================================================================
# Cantidades del portafolio real en HowTheMarketWorks (negativo = venta en corto).
# Posiciones previas más las compras de NVDA, GOOGL, BAC, CVX y COP del 9 de noviembre de 2026.
HTMW = {"XOM": 1316, "MSFT": 470, "JPM": 602, "AAPL": 730, "USO": 170, "GS": -55,
        "NVDA": 62, "GOOGL": 42, "BAC": 242, "CVX": 56, "COP": 88}

H = 60
tr, te = r.iloc[:-H], r.iloc[-H:]
vols, zres = {}, {}
for k in ACTIVOS:
    m = arch_model(tr[k], mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    vols[k] = np.sqrt(m.forecast(horizon=H).variance.iloc[-1].values.mean())
    zres[k] = m.std_resid
vols = pd.Series(vols)
R = pd.DataFrame(zres).corr()
Sigma = np.diag(vols.values) @ R.values @ np.diag(vols.values)
var_p = lambda w: float(w @ Sigma @ w)

n = len(ACTIVOS)
w_eq = np.repeat(1 / n, n)
w_mv = minimize(var_p, w_eq, method="SLSQP", bounds=[(0, 1)] * n,
                constraints={"type": "eq", "fun": lambda w: w.sum() - 1}).x
valor = pd.Series(HTMW).reindex(ACTIVOS).fillna(0) * precios.iloc[-H - 1][ACTIVOS]
w_real = (valor / valor.sum()).values

pesos = pd.DataFrame({"Iguales (1/N)": w_eq, "Mínima varianza (GARCH)": w_mv, "Real (HowTheMarketWorks)": w_real}, index=ACTIVOS)
display((100 * pesos).round(1).rename(columns=lambda c: c + " (%)"))

res = {}
for nombre, w in pesos.items():
    rp = te @ w.values
    sig_ = np.sqrt(var_p(w.values))
    res[nombre] = {"Vol. pronosticada anual (%)": sig_ * np.sqrt(252), "Vol. realizada anual (%)": rp.std() * np.sqrt(252),
                   "VaR 95% 1 día (%)": 1.645 * sig_, "Rend. acumulado fuera de muestra (%)": (np.exp(rp.sum() / 100) - 1) * 100,
                   "Peor día (%)": rp.min()}
display(pd.DataFrame(res).round(2))
pd.DataFrame({nombre: np.exp((te @ w.values).cumsum() / 100) for nombre, w in pesos.items()}, index=te.index).plot(
    title="Valor de $1 invertido (60 días fuera de muestra)"); plt.show()


# ======================================================================
# 9.2 Simulación de escenarios (Monte Carlo)
# ======================================================================
rng = np.random.default_rng(2026)
N_SIM, DIAS, V0 = 10_000, 60, 1_000_000
mu = r.mean().values
Lc = np.linalg.cholesky(Sigma + 1e-10 * np.eye(n))
sim = {}
for nombre, w in pesos.items():
    z = rng.standard_normal((N_SIM, DIAS, n)) @ Lc.T + mu
    rp = z @ w.values / 100
    sim[nombre] = V0 * np.exp(np.cumsum(rp, axis=1))

escen = pd.DataFrame({nombre: {"P5 (pesimista)": np.percentile(v[:, -1], 5), "Mediana": np.median(v[:, -1]),
                               "P95 (optimista)": np.percentile(v[:, -1], 95),
                               "Prob. de pérdida (%)": 100 * np.mean(v[:, -1] < V0),
                               "VaR 95% 60 días (USD)": V0 - np.percentile(v[:, -1], 5)} for nombre, v in sim.items()})
display(escen.round(1))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for ax, (nombre, v) in zip(axes, sim.items()):
    for qq, a in [(5, 0.15), (25, 0.3)]:
        ax.fill_between(range(1, DIAS + 1), np.percentile(v, qq, axis=0), np.percentile(v, 100 - qq, axis=0), alpha=a, color="tab:blue")
    ax.plot(range(1, DIAS + 1), np.median(v, axis=0), "k", lw=1.5)
    ax.axhline(V0, color="r", ls="--", lw=1); ax.set_title(nombre); ax.set_xlabel("Días hábiles")
axes[0].set_ylabel("Valor del portafolio (USD)")
plt.suptitle("Escenarios simulados: mediana, bandas 25–75% y 5–95%"); plt.tight_layout(); plt.show()


# ======================================================================
# 10. Análisis técnico: MACD y RSI
# ======================================================================

# ======================================================================
# 10.1 Señales al cierre del periodo
# ======================================================================
def macd(p):
    m = p.ewm(span=12, adjust=False).mean() - p.ewm(span=26, adjust=False).mean()
    return m, m.ewm(span=9, adjust=False).mean()

def rsi(p, n=14):
    d = p.diff()
    ganan = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    pierde = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + ganan / pierde)

filas = []
for k in ACTIVOS:
    m, s = macd(precios[k]); rr = rsi(precios[k])
    cruce = np.sign(m - s).diff().fillna(0)
    ult = cruce[cruce != 0].index[-1].date() if (cruce != 0).any() else None
    filas.append({"Activo": k, "RSI(14)": rr.iloc[-1],
                  "Zona RSI": "Sobrecompra" if rr.iloc[-1] > 70 else "Sobreventa" if rr.iloc[-1] < 30 else "Neutral",
                  "MACD": m.iloc[-1], "Señal": s.iloc[-1], "Tendencia MACD": "Alcista" if m.iloc[-1] > s.iloc[-1] else "Bajista",
                  "Último cruce": ult})
tecnico = pd.DataFrame(filas).set_index("Activo")
display(tecnico.round(2))


# ======================================================================
# 10.2 Prueba retrospectiva (*backtesting*) de la estrategia MACD
# ======================================================================
def metricas(rd):
    anual = rd.mean() * 252; vol = rd.std() * np.sqrt(252)
    nivel = np.exp(rd.cumsum() / 100)
    return anual, vol, (anual - rf_anual) / vol, (nivel / nivel.cummax() - 1).min() * 100

filas = []
for k in ACTIVOS:
    m, s = macd(precios[k])
    pos = (m > s).astype(int).shift(1).reindex(r.index).fillna(0)
    bh, st = metricas(r[k]), metricas(r[k] * pos)
    filas.append({"Activo": k, "B&H rend. anual (%)": bh[0], "MACD rend. anual (%)": st[0],
                  "B&H Sharpe": bh[2], "MACD Sharpe": st[2], "B&H máx. caída (%)": bh[3], "MACD máx. caída (%)": st[3],
                  "Días invertido (%)": 100 * pos.mean(), "Operaciones": int(pos.diff().abs().sum())})
bt = pd.DataFrame(filas).set_index("Activo")
display(bt.round(2))
print(f"MACD supera a B&H en Sharpe en {(bt['MACD Sharpe'] > bt['B&H Sharpe']).sum()} de {len(bt)} activos.")


# ======================================================================
# 11. Estrategia y propuesta a Capital Analyst
# ======================================================================

# ======================================================================
# 11.1 Horizonte, perfil de riesgo y universo
# ======================================================================

# ======================================================================
# 11.2 Estrategia propuesta
# ======================================================================

# ======================================================================
# 11.3 Situación actual (6 de noviembre de 2026)
# ======================================================================

# ======================================================================
# 12. Evidencias: operaciones en HowTheMarketWorks
# ======================================================================

# ======================================================================
# 13. Conclusiones
# ======================================================================

# ======================================================================
# 14. Referencias
# ======================================================================