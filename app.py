"""
DCA / Martingale-jellegű FUTURES stratégia backtesztelő - Streamlit felület
==============================================================================
Teljesen paraméterezhető tőkeáttételes (leverage) backtesztelő, valós
számlaszimulációval (cross margin egyenleg, margin-foglalás, egyszerűsített
likvidáció). Nem kell hozzá programozás - a mezőket kitöltöd, gombot nyomsz.

FONTOS EGYSZERŰSÍTÉSEK (nem helyettesíti a valós tőzsdei végrehajtást):
  - Nincs benne kereskedési díj (maker/taker fee)
  - Nincs benne funding rate (a perpetual futures-nél ez időszakosan
    fizetendő/kapható díj - hosszú backtesztnél ez számottevő lehet)
  - A likvidáció egy EGYSZERŰSÍTETT közelítés (nem veszi figyelembe a
    tőzsde pontos maintenance margin táblázatát)
  - Nincs csúszás (slippage)
  Csak tájékoztató / stratégia-összehasonlító célra való, NEM befektetési
  tanács, és a valós kereskedési eredmény ettől eltérhet.
"""

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"

INTERVAL_MAP = {"1m": "1", "5m": "5", "15m": "15", "1h": "60"}

st.set_page_config(page_title="DCA Futures Backtest", layout="wide")
st.title("📊 DCA / Martingale Futures Stratégia Backtesztelő")
st.caption("Tőkeáttételes, teljesen paraméterezhető backtest - válaszd ki az eszközöket és a beállításokat.")

ALL_ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "BNB": "BNBUSDT",
    "SUI": "SUIUSDT", "XRP": "XRPUSDT", "TRON (TRX)": "TRXUSDT", "HYPE": "HYPEUSDT",
    "XMR": "XMRUSDT", "DOGE": "DOGEUSDT", "ADA": "ADAUSDT", "LINK": "LINKUSDT",
    "XLM": "XLMUSDT", "XAUT (arany)": "XAUTUSDT", "XAG (ezüst)": "XAGUSDT",
}

# ---------------------------------------------------------------------
# ADATLETÖLTÉS (Binance Futures - USDT-M Perpetual)
# ---------------------------------------------------------------------

def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, status=None) -> pd.DataFrame:
    bybit_interval = INTERVAL_MAP.get(interval, "5")
    all_rows = []
    cur = start_ms
    while cur < end_ms:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": bybit_interval,
            "start": cur,
            "end": end_ms,
            "limit": 1000,
        }
        resp = requests.get(BYBIT_KLINE_URL, params=params, timeout=15)
        if resp.status_code != 200:
            if status:
                status.warning(f"⚠️ {symbol}: nem sikerült adatot lekérni ({resp.status_code}).")
            break
        payload = resp.json()
        if payload.get("retCode") != 0:
            if status:
                status.warning(f"⚠️ {symbol}: {payload.get('retMsg', 'ismeretlen hiba')} "
                                f"(lehet, hogy ez a szimbólum nem elérhető a Bybit-en).")
            break
        rows = payload.get("result", {}).get("list", [])
        if not rows:
            break
        # A Bybit legújabb->legrégebbi sorrendben adja vissza -> fordítsuk meg
        rows = sorted(rows, key=lambda r: int(r[0]))
        # Ha ugyanazt a szakaszt kaptuk vissza (nincs több új adat), álljunk le
        last_ts = int(rows[-1][0])
        if last_ts <= cur:
            break
        all_rows.extend(rows)
        cur = last_ts + 1
        if status:
            progress_pct = min(100, int((cur - start_ms) / max(1, end_ms - start_ms) * 100))
            status.info(f"⏳ {symbol} adatok letöltése... ({progress_pct}%)")
        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "turnover"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.drop_duplicates(subset="open_time")
    return df[["open_time", "open", "high", "low", "close"]]


def fetch_asset_history(symbol: str, months_back: int, interval: str, status=None) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=30 * months_back)
    return fetch_klines(symbol, interval, int(start.timestamp() * 1000), int(end.timestamp() * 1000), status)


# ---------------------------------------------------------------------
# SZIMULÁCIÓS MOTOR (leverage + margin-tudatos)
# ---------------------------------------------------------------------

LIQUIDATION_MARGIN_USAGE_PCT = 90  # a margin ennyi %-ának elvesztésekor tekintjük likvidáltnak (egyszerűsítés)

class Trade:
    def __init__(self, open_time, open_price, entry_targets, tp_pct, trail_trigger,
                 trail_pct, leverage, allocated_margin):
        self.open_time = open_time
        self.entry_targets = entry_targets
        self.tp_pct = tp_pct
        self.trail_trigger = trail_trigger
        self.trail_pct = trail_pct
        self.leverage = leverage
        self.allocated_margin = allocated_margin  # ennyi USDT margint köt le a számlán
        self.entries_filled = [(entry_targets[0][0], open_price)]
        self.avg_price = open_price
        self.highest_price = open_price
        self.closed = False
        self.close_reason = None
        self.close_time = None
        self.pnl_pct_price = None   # az árfolyam %-os elmozdulása (nem margin-arányos)
        self.pnl_usdt = None        # a tényleges USDT eredmény (margin + leverage figyelembevételével)

    def update_avg_price(self):
        total_w = sum(w for w, _ in self.entries_filled)
        self.avg_price = sum(w * p for w, p in self.entries_filled) / total_w

    def _finalize(self, reason, close_time, pnl_pct_price):
        self.closed = True
        self.close_reason = reason
        self.close_time = close_time
        self.pnl_pct_price = pnl_pct_price
        margin_pnl_pct = pnl_pct_price * self.leverage
        margin_pnl_pct = max(margin_pnl_pct, -100.0)  # nem veszíthetsz többet, mint a margin 100%-a
        self.pnl_usdt = self.allocated_margin * margin_pnl_pct / 100

    def step(self, candle):
        if self.closed:
            return
        low, high = candle.low, candle.high
        self.highest_price = max(self.highest_price, high)

        # 1) Újabb DCA szint teljesülése
        filled_so_far = len(self.entries_filled)
        if filled_so_far < len(self.entry_targets):
            next_weight, next_dev = self.entry_targets[filled_so_far]
            target_price = self.entries_filled[0][1] * (1 + next_dev / 100)
            if low <= target_price:
                self.entries_filled.append((next_weight, target_price))
                self.update_avg_price()

        # 2) Likvidáció ellenőrzése (a jelenlegi átlagár és a leverage alapján, egyszerűsítve)
        adverse_move_pct = (self.avg_price - low) / self.avg_price * 100
        if adverse_move_pct * self.leverage >= LIQUIDATION_MARGIN_USAGE_PCT:
            self._finalize("LIQUIDATED", candle.open_time, -adverse_move_pct)
            return

        # 3) Take-Profit
        tp_price = self.avg_price * (1 + self.tp_pct / 100)
        if high >= tp_price:
            self._finalize("TP", candle.open_time, self.tp_pct)
            return

        # 4) Trailing Stop
        drawdown_from_high = (self.highest_price - low) / self.highest_price * 100
        if drawdown_from_high >= self.trail_trigger:
            stop_price = self.highest_price * (1 - self.trail_pct / 100)
            if low <= stop_price:
                pnl_pct_price = (stop_price - self.avg_price) / self.avg_price * 100
                self._finalize("TRAILING_STOP", candle.open_time, pnl_pct_price)
                return


def run_backtest(df, asset_name, entry_targets, tp_pct, trail_trigger, trail_pct,
                  interval_minutes, max_simultaneous, leverage,
                  trade_value_mode, trade_value_usdt, trade_value_pct, start_balance):

    if df.empty or len(df) < 10:
        return {"Eszköz": asset_name, "Hiba": "Nincs elérhető adat ehhez a szimbólumhoz"}, None

    df = df.sort_values("open_time").reset_index(drop=True)
    open_trades, closed_trades = [], []
    last_open_time = None
    balance = start_balance
    peak_balance = start_balance
    max_drawdown_pct = 0.0
    skipped_no_margin = 0
    liquidation_count = 0
    balance_history = []

    for _, candle in df.iterrows():
        used_margin = sum(t.allocated_margin for t in open_trades)
        available_margin = balance - used_margin

        should_try_open = (
            (last_open_time is None or
             (candle.open_time - last_open_time) >= timedelta(minutes=interval_minutes))
            and len(open_trades) < max_simultaneous
        )

        if should_try_open:
            if trade_value_mode == "Fix USDT":
                trade_value = trade_value_usdt
            else:
                trade_value = balance * (trade_value_pct / 100)

            margin_needed = trade_value / leverage

            if margin_needed <= available_margin and margin_needed > 0:
                open_trades.append(Trade(
                    candle.open_time, candle.open, entry_targets, tp_pct,
                    trail_trigger, trail_pct, leverage, margin_needed
                ))
                last_open_time = candle.open_time
            else:
                skipped_no_margin += 1
                last_open_time = candle.open_time  # ne próbálkozzon minden gyertyán újra

        for t in open_trades:
            t.step(candle)

        newly_closed = [t for t in open_trades if t.closed]
        for t in newly_closed:
            balance += t.pnl_usdt
            if t.close_reason == "LIQUIDATED":
                liquidation_count += 1
        closed_trades.extend(newly_closed)
        open_trades = [t for t in open_trades if not t.closed]

        peak_balance = max(peak_balance, balance)
        if peak_balance > 0:
            dd = (peak_balance - balance) / peak_balance * 100
            max_drawdown_pct = max(max_drawdown_pct, dd)

        balance_history.append({"time": candle.open_time, "balance": balance})

    n_closed = len(closed_trades)
    if n_closed == 0:
        return {"Eszköz": asset_name, "Hiba": "Egyetlen trade sem zárult le (túl kevés margin vagy adat?)"}, None

    wins = [t for t in closed_trades if t.pnl_usdt > 0]
    total_pnl_usdt = sum(t.pnl_usdt for t in closed_trades)

    summary = {
        "Eszköz": asset_name,
        "Lezárt trade-ek": n_closed,
        "Nyitva maradt": len(open_trades),
        "Likvidációk száma": liquidation_count,
        "Kihagyva (nincs margin)": skipped_no_margin,
        "Nyerő arány (%)": round(len(wins) / n_closed * 100, 2),
        "Végső egyenleg (USDT)": round(balance, 2),
        "Teljes hozam (%)": round((balance - start_balance) / start_balance * 100, 2),
        "Max. drawdown (%)": round(max_drawdown_pct, 2),
        "Összesített PnL (USDT)": round(total_pnl_usdt, 2),
    }
    return summary, pd.DataFrame(balance_history)


# ---------------------------------------------------------------------
# FELÜLET - BEMENETI MEZŐK
# ---------------------------------------------------------------------

st.header("1️⃣ Eszközök kiválasztása")
selected_assets = st.multiselect(
    "Milyen eszközökre fusson a backtest?",
    options=list(ALL_ASSETS.keys()),
    default=["BTC", "ETH", "SOL"],
)
st.caption("Az adatok forrása: Bybit nyilvános API (linear/USDT perpetual futures) - "
           "ugyanaz a tőzsde, ahol ténylegesen kereskedsz.")
st.caption("Megjegyzés: HYPE, XMR és XAG esetében előfordulhat, hogy csak korlátozott "
           "múltra visszamenőleg (pl. listázás óta) van adat. Az XAG (ezüst) esetében ez "
           "különösen igaz, mivel a kontraktus csak 2026 január eleje óta létezik.")

st.header("2️⃣ Számla és pozícióméretezés")
col_a, col_b, col_c = st.columns(3)
with col_a:
    start_balance = st.number_input("Cross margin számlaegyenleg (USDT)", min_value=1.0, value=1000.0, step=100.0)
    leverage = st.number_input("Tőkeáttétel (leverage, x)", min_value=1, max_value=125, value=20)
with col_b:
    trade_value_mode = st.radio("Egy trade mérete", ["Fix USDT", "% a margin egyenlegből"])
with col_c:
    if trade_value_mode == "Fix USDT":
        trade_value_usdt = st.number_input("Fix tétel (USDT, notional/pozícióérték)", min_value=1.0, value=100.0)
        trade_value_pct = 0.0
    else:
        trade_value_pct = st.number_input("Tétel a mindenkori egyenleg %-ában", min_value=0.1, value=2.0, step=0.1)
        trade_value_usdt = 0.0

st.caption(
    "A 'trade mérete' a pozíció NOTIONAL (teljes) értéke, a tőkeáttétellel elosztva adja a lekötött margint "
    "(pl. 100 USDT tétel, 20x leverage → 5 USDT margint köt le trade-enként)."
)

st.header("3️⃣ Időzítés és párhuzamos trade-ek")
col_d, col_e = st.columns(2)
with col_d:
    trade_interval_minutes = st.selectbox("Új trade nyitása ennyi percenként", [5, 10, 15], index=1)
    interval = st.selectbox("Adatfelbontás (candle timeframe a szimulációhoz)", ["1m", "5m", "15m", "1h"], index=1)
with col_e:
    max_simultaneous = st.number_input("Max. egyidejű trade-ek száma", min_value=1, value=15)
    months_back = st.number_input("Hány hónapra visszamenőleg?", min_value=1, max_value=60, value=24)

st.header("4️⃣ Take-Profit és Trailing Stop")
col_f, col_g, col_h = st.columns(3)
with col_f:
    tp_pct = st.number_input("Take-Profit (%, átlagár felett)", min_value=0.01, value=0.61, step=0.01, format="%.2f")
with col_g:
    trail_trigger = st.number_input("Trailing Trigger (%, csúcs alatt)", min_value=0.01, value=0.5, step=0.01)
with col_h:
    trail_pct = st.number_input("Trailing Percent (%)", min_value=0.01, value=0.4, step=0.01)

st.header("5️⃣ DCA (Entry) szintek")
n_levels = st.number_input("Hány DCA szint legyen?", min_value=1, max_value=15, value=6)

default_targets = [
    (1.07, 0.00), (2.32, -7.38), (5.06, -15.00),
    (11.15, -22.69), (24.76, -30.39), (55.64, -38.01),
]

entry_targets = []
n_cols = min(int(n_levels), 6)
level_rows = [st.columns(n_cols) for _ in range((int(n_levels) + n_cols - 1) // n_cols)]
flat_cols = [c for row in level_rows for c in row]

for i in range(int(n_levels)):
    default_w, default_d = default_targets[i] if i < len(default_targets) else (0.0, -5.0 * (i + 1))
    with flat_cols[i]:
        st.markdown(f"**Szint {i+1}**")
        w = st.number_input(f"Súly % ({i+1})", value=float(default_w), key=f"w{i}", format="%.2f")
        d = st.number_input(f"Deviáció % ({i+1})", value=float(default_d), key=f"d{i}", format="%.2f")
        entry_targets.append((w, d))

total_weight = sum(w for w, _ in entry_targets)
if abs(total_weight - 100) > 0.5:
    st.warning(f"⚠️ A szintek súlyának összege {total_weight:.2f}%, nem 100% - érdemes ellenőrizni.")
else:
    st.success(f"✅ A szintek súlyának összege: {total_weight:.2f}%")

st.divider()

# ---------------------------------------------------------------------
# FUTTATÁS
# ---------------------------------------------------------------------

if st.button("🚀 Backtest futtatása", type="primary"):
    if not selected_assets:
        st.error("Válassz ki legalább egy eszközt!")
    else:
        results = []
        balance_curves = {}
        overall_progress = st.progress(0.0, text="Indítás...")

        for idx, asset_label in enumerate(selected_assets):
            symbol = ALL_ASSETS[asset_label]
            status = st.empty()

            df = fetch_asset_history(symbol, months_back, interval, status)
            status.info(f"⚙️ Szimuláció futtatása: {asset_label}...")

            summary, balance_df = run_backtest(
                df, asset_label, entry_targets, tp_pct, trail_trigger, trail_pct,
                trade_interval_minutes, max_simultaneous, leverage,
                trade_value_mode, trade_value_usdt, trade_value_pct, start_balance
            )
            results.append(summary)
            if balance_df is not None:
                balance_curves[asset_label] = balance_df
            status.empty()
            overall_progress.progress((idx + 1) / len(selected_assets), text=f"{asset_label} kész")

        overall_progress.empty()

        st.header("✅ Eredmények")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Eredmények letöltése CSV-ben", csv, "futures_backtest_eredmenyek.csv", "text/csv")

        if balance_curves:
            st.header("📈 Egyenleg alakulása időben")
            chosen = st.selectbox("Melyik eszköz egyenleggörbéjét nézzük?", list(balance_curves.keys()))
            chart_df = balance_curves[chosen].set_index("time")
            st.line_chart(chart_df["balance"])

        st.caption(
            "⚠️ Egyszerűsített szimuláció: nincs benne díj, funding rate és csúszás, "
            "a likvidáció-számítás közelítő jellegű. Csak tájékoztató, nem befektetési tanács."
        )
