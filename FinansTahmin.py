import requests
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from statsmodels.tsa.statespace.sarimax import SARIMAX

url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {
    "vs_currency": "try",
    "days": "180",
    "interval": "daily",
}

response = requests.get(url, params=params)
veri = response.json()

fiyatlar = veri["prices"]
df = pd.DataFrame(fiyatlar, columns=["zaman_damgasi", "price"])
df["zaman_damgasi"] = pd.to_datetime(df["zaman_damgasi"], unit="ms")
df.set_index("zaman_damgasi", inplace=True)

df["price"] = pd.to_numeric(df["price"], errors="coerce")

print(f"Number of NaN or invalid values: {df.isnull().sum()}")

df = df.asfreq("D")

model = SARIMAX(df["price"], order=(2,1,2), seasonal_order=(1,1,1,30))
model_fit = model.fit()

tahmin = model_fit.forecast(steps=30)
tahmin_indeksi = pd.date_range(start=df.index[-12] + pd.Timedelta(days=1), periods=30)
tahmin_serisi = pd.Series(tahmin, index=tahmin_indeksi)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["price"],
    mode="lines",
    name="Bitcoin price data in TRY",
    line=dict(color="blue"),
))

fig.add_trace(go.Scatter(
    x=tahmin_serisi.index,
    y=tahmin_serisi,
    mode="lines",
    name="30-day Bitcoin forecast",
    line=dict(color="red", dash="dash")
))

fig.update_layout(
    title="Bitcoin prices – last 6 months",
    xaxis_title="Date",
    yaxis_title="Price (TRY)",
    hovermode="x",
    height=600,
    template="plotly_dark"
)

pio.renderers.default = 'browser'
fig.show()
