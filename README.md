# FZ4038 – Proyecto Final – Modelos – Capital Analyst

**Automatización de algoritmos para el portafolio de Capital Analyst: regresión, datos panel, ARIMA, GARCH y simulación**

- **Curso:** Econometría Financiera (FZ4038) · EGADE Business School, Tecnológico de Monterrey
- **Profesor:** Dr. Raymundo Díaz Robles
- **Alumno:** Daniel Mondragón Monroy (A01452205)
- **Fecha:** Noviembre de 2026

## El caso

Capital Analyst es una firma mexicana de asesoría de inversiones que automatiza su análisis con algoritmos en R y Python. En este proyecto se actúa como analista de la firma: se construye, en un solo cuaderno reproducible, el flujo completo que propone el caso, desde la obtención de datos hasta una propuesta de inversión.

## Contenido del repositorio

| Archivo | Descripción |
|:--|:--|
| `Proyecto_Final_Capital_Analyst_A01452205_FZ4038.html` | Documento final con código, resultados, gráficas e interpretaciones. Se abre en cualquier navegador. |
| `Proyecto_Final_Capital_Analyst_A01452205_FZ4038.ipynb` | Notebook de Python (Google Colab) con los resultados de la corrida. |
| `Proyecto_Final_Capital_Analyst_A01452205_FZ4038_codigo.py` | Solo el código del notebook, sin resultados, ordenado por secciones. Se usó para la revisión con GitHub Copilot. |
| `panel_trimestral_LSEG.csv` | Datos trimestrales depurados para el modelo de datos panel; el notebook los lee desde este repositorio. |
| `Proyecto_Final_Datos_Panel_LSEG.xlsx` | Descarga original de LSEG Workspace (estados financieros trimestrales), fuente del archivo CSV. |

## Universo de inversión

Las 10 acciones del portafolio en HowTheMarketWorks y el ETF USO, agrupados por sector:

| Sector | Activos |
|:--|:--|
| Tecnología | AAPL, MSFT, NVDA, GOOGL |
| Bancos | JPM, GS, BAC |
| Energía | XOM, CVX, COP |
| ETF de petróleo | USO |

## Datos

- **Precios diarios** (2 de enero de 2015 al 6 de noviembre de 2026): Yahoo Finance, mediante la librería `yfinance`.
- **Factores de mercado:** S&P 500, tasa del Tesoro a 10 años, petróleo Brent, índice dólar y T-Bill a 3 meses (Yahoo Finance).
- **Reportes trimestrales** (2015–2026): ingresos, utilidad neta de la controladora, capital contable común y activo total, de LSEG Workspace. Los trimestres fiscales de Apple y NVIDIA se alinean con el trimestre calendario.

## Estructura del análisis

1. **El caso Capital Analyst y objetivo**
2. **Universo de inversión y datos:** fuentes, variables y calidad de los datos.
3. **Métricas de riesgo y rendimiento:** rendimiento y volatilidad anualizados, índice de Sharpe, caída máxima, VaR y CVaR, correlaciones.
4. **Regresión lineal simple:** modelo de mercado (beta de cada acción frente al S&P 500), con pruebas de supuestos y errores robustos de Newey-West.
5. **Regresión lineal múltiple:** modelo de factores (mercado, tasas, petróleo y dólar) con justificación de las variables por sector, VIF y pruebas.
6. **Datos panel:** ¿qué fundamentales de los reportes trimestrales explican el rendimiento de las acciones? Modelos agrupado, de efectos fijos y aleatorios; pruebas F, de Hausman, de Wooldridge y de Pesaran; errores de Driscoll-Kraay.
7. **Series de tiempo (ARIMA):** metodología Box-Jenkins, pronóstico fuera de muestra frente al pronóstico ingenuo y pronóstico a 20 días.
8. **Volatilidad (GARCH):** GARCH(1,1) con distribución t y GJR-GARCH (efecto apalancamiento).
9. **Portafolio:** covarianza pronosticada con GARCH, portafolios de pesos iguales, de mínima varianza y el portafolio real, y simulación Monte Carlo de escenarios.
10. **Análisis técnico:** señales MACD y RSI y prueba retrospectiva (*backtesting*).
11. **Estrategia y propuesta a Capital Analyst**
12. **Evidencias:** operaciones en HowTheMarketWorks.
13. **Conclusiones**
14. **Referencias**

Los resultados y su interpretación están en el archivo HTML.

## Cómo reproducir

1. Abrir el archivo `.ipynb` en [Google Colab](https://colab.research.google.com) (**File → Upload notebook**).
2. Ejecutar **Runtime → Run all**. El notebook instala las librerías necesarias (`yfinance`, `arch`, `linearmodels`), descarga los precios y lee el archivo CSV de este repositorio. La fecha final está fija para que los resultados sean reproducibles. La corrida completa tarda entre 5 y 10 minutos.

## Uso de inteligencia artificial

Para elaborar el código en Python y apoyar la redacción se utilizó Claude (Anthropic) como asistente, y GitHub Copilot para la revisión del código ([conversación con Copilot](https://github.com/copilot/share/c04d412c-01c4-8ea6-9902-9a04c0db6916)). Las observaciones de Copilot que se consideraron válidas se incorporaron como advertencias y limitaciones en las secciones 6, 9 y 13. Los cálculos se ejecutaron en Google Colab y los resultados e interpretaciones fueron revisados por el alumno.
