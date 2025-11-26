# regime_adaptive.py
# Regime-Adaptive Freqtrade Strategy
# - Detect market regime on higher timeframes (1h, 4h)
# - Switch behavior:
#     UP   -> trend-following long (pullback entries)
#     DOWN -> short trend-following (pullback entries)
#     SIDE -> grid/mean-reversion style entries (BB lower -> buy, BB upper -> sell)
#
# Author: Senior Python dev (commented & ready for backtest)
# Notes:
#  - Requires 'technical' package (qtpylib) + ta (talib.abstract) installed in the container.
#  - Put this file in user_data/strategies/ and run backtest/trade as usual.
#  - Tune hyperparams (ROI/stoploss/grid sizes) for your capital and fee model.

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np
import talib.abstract as ta
from technical import qtpylib

class RegimeAdaptiveStrategy(IStrategy):
    """
    Regime-Adaptive strategy:
    - Uses 1h and 4h informatives to determine regime
    - Uses indicators: EMA50/200, ADX, ATR, Bollinger Bands, MACD, RSI
    - Modes:
        * UP   : long-only, enter on pullback to ema20 or MACD cross
        * DOWN : short-only, enter on pullback to ema20 or MACD cross
        * SIDE : mean-reversion / grid-friendly entries using BBands
    """

    INTERFACE_VERSION = 3
    # Allow short — set to True if you run futures/dry_run that supports short.
    can_short = False

    # Primary trading timeframe (fast execution / signal granularity)
    timeframe = "5m"

    # Informative timeframes for regime detection (higher TF reduces whipsaw)
    informative_timeframes = ["1h", "4h"]

    # Risk management baseline
    minimal_roi = {"0": 0.02}   # default ROI target (can be tuned)
    stoploss = -0.10            # fallback stoploss

    # Trailing stop (useful in uptrends)
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    # candles needed
    startup_candle_count = 400  # enough to compute 200-EMA on 4h

    # ----- Hyperparameters you can tune or expose to hyperopt -----
    adx_threshold = 20        # ADX threshold to qualify as trending
    atr_mult_low = 0.5        # if ATR low, widen grid spacing or pause
    bb_window = 20
    bb_std = 2
    rsi_oversold = 30
    rsi_overbought = 70
    pullback_ema = 20         # pullback EMA for entries
    grid_bb_percent = 0.35    # percent from BB middle to set grid steps (approx)

    # ---------------------------------------------------------------
    def informative_pairs(self):
        """
        Return pairs/timeframes to be cached: for each pair we want 1h and 4h
        """
        pairs = self.dp.current_whitelist()
        pairs_tf = []
        for p in pairs:
            for tf in self.informative_timeframes:
                pairs_tf.append((p, tf))
        return pairs_tf

    def _calc_informative_indicators(self, df: DataFrame) -> DataFrame:
        """
        Helper: compute common indicators on an informative dataframe
        (expected to be 1h or 4h)
        """
        res = df.copy()
        res["ema50"] = ta.EMA(res, timeperiod=50)
        res["ema200"] = ta.EMA(res, timeperiod=200)
        res["adx"] = ta.ADX(res)
        res["atr"] = ta.ATR(res, timeperiod=14)
        return res

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculates indicators for the primary timeframe (5m).
        Also merges informative (1h,4h) via dp.get_pair_dataframe in regime checks.
        """
        df = dataframe

        # Primary TF indicators (fast)
        df["ema20"] = ta.EMA(df, timeperiod=20)
        df["ema50"] = ta.EMA(df, timeperiod=50)
        df["ema200"] = ta.EMA(df, timeperiod=200)
        macd = ta.MACD(df)
        df["macd"] = macd["macd"]
        df["macdsignal"] = macd["macdsignal"]
        df["rsi"] = ta.RSI(df, timeperiod=14)
        df["atr"] = ta.ATR(df, timeperiod=14)

        # Bollinger Bands for mean-reversion/grid decisions
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=self.bb_window, stds=self.bb_std)
        df["bb_lowerband"] = bb["lower"]
        df["bb_middleband"] = bb["mid"]
        df["bb_upperband"] = bb["upper"]
        df["bb_width"] = (df["bb_upperband"] - df["bb_lowerband"]) / df["bb_middleband"]

        return df

    # ---------------------------
    # Regime detection helpers
    # ---------------------------
    def is_uptrend(self, pair: str) -> bool:
        """
        Define uptrend using 1h and 4h EMA50/EMA200 and ADX strength.
        Returns True if both higher TFs indicate bullish regime.
        """
        try:
            df1h = self.dp.get_pair_dataframe(pair, "1h")
            df4h = self.dp.get_pair_dataframe(pair, "4h")

            inf1 = self._calc_informative_indicators(df1h)
            inf4 = self._calc_informative_indicators(df4h)

            # Conditions: EMA50 > EMA200 on both TFs, price above EMA50 on 1h, ADX indicates trend
            cond1 = (inf1["ema50"].iloc[-1] > inf1["ema200"].iloc[-1]) and (inf1["close"].iloc[-1] > inf1["ema50"].iloc[-1])
            cond2 = (inf4["ema50"].iloc[-1] > inf4["ema200"].iloc[-1])
            adx_ok = (inf1["adx"].iloc[-1] >= self.adx_threshold) or (inf4["adx"].iloc[-1] >= self.adx_threshold)

            return cond1 and cond2 and adx_ok
        except Exception:
            # If informative not available, fallback conservative local check
            return False

    def is_downtrend(self, pair: str) -> bool:
        """
        Define downtrend mirrored to is_uptrend.
        """
        try:
            df1h = self.dp.get_pair_dataframe(pair, "1h")
            df4h = self.dp.get_pair_dataframe(pair, "4h")

            inf1 = self._calc_informative_indicators(df1h)
            inf4 = self._calc_informative_indicators(df4h)

            cond1 = (inf1["ema50"].iloc[-1] < inf1["ema200"].iloc[-1]) and (inf1["close"].iloc[-1] < inf1["ema50"].iloc[-1])
            cond2 = (inf4["ema50"].iloc[-1] < inf4["ema200"].iloc[-1])
            adx_ok = (inf1["adx"].iloc[-1] >= self.adx_threshold) or (inf4["adx"].iloc[-1] >= self.adx_threshold)

            return cond1 and cond2 and adx_ok
        except Exception:
            return False

    def is_sideway(self, pair: str) -> bool:
        """
        Sideway: not uptrend and not downtrend on higher TFs, and BB width on 1h is moderate.
        """
        # if neither up nor down, consider sideway; additional check with BB width
        try:
            if self.is_uptrend(pair) or self.is_downtrend(pair):
                return False
            df1h = self.dp.get_pair_dataframe(pair, "1h")
            bb = qtpylib.bollinger_bands(qtpylib.typical_price(df1h), window=self.bb_window, stds=self.bb_std)
            bb_width = (bb["upper"].iloc[-1] - bb["lower"].iloc[-1]) / bb["mid"].iloc[-1]
            # sideway if bb_width within reasonable window (not extremely low, not extremely high)
            return (bb_width > 0.005) and (bb_width < 0.06)
        except Exception:
            # fallback: default to False to be conservative
            return False

    # ---------------------------
    # Entry & exit population
    # ---------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entries depend on detected regime. We compute the regime once per pair and apply rules.
        Important: dataframes are vectorized; we set conditions for rows where signals should be active.
        """
        pair = metadata["pair"]

        # By default no signals
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        # Determine regime once (fast boolean)
        up = self.is_uptrend(pair)
        down = self.is_downtrend(pair)
        side = self.is_sideway(pair)

        # ----- UPTREND mode (long-only)
        if up:
            # Entry idea: pullback to EMA20 OR MACD crossover up
            dataframe.loc[
                (
                    # price near or slightly below ema20 (pullback)
                    (dataframe["close"] <= dataframe["ema20"] * 1.002)
                    & (dataframe["rsi"] > 30)  # avoid oversold (too deep)
                    & (dataframe["macd"] > dataframe["macdsignal"])  # bullish signal
                ),
                "enter_long",
            ] = 1

            # Also accept MACD cross even if price not near ema20 (momentum entries)
            dataframe.loc[
                (
                    qtpylib.crossed_above(dataframe["macd"], dataframe["macdsignal"])
                    & (dataframe["rsi"] > 35)
                ),
                "enter_long",
            ] = 1

        # ----- DOWNTREND mode (short-first)
        if down:
            # Entry idea: on pullback to ema20 (price moves up to ema20 then resumes down)
            dataframe.loc[
                (
                    (dataframe["close"] >= dataframe["ema20"] * 0.998)
                    & (dataframe["macd"] < dataframe["macdsignal"])   # bearish momentum
                    & (dataframe["rsi"] < 80)
                ),
                "enter_short",
            ] = 1

            # Also short on MACD crossing below
            dataframe.loc[
                (
                    qtpylib.crossed_below(dataframe["macd"], dataframe["macdsignal"])
                    & (dataframe["rsi"] < 70)
                ),
                "enter_short",
            ] = 1

        # ----- SIDEWAY mode (grid-friendly / mean-reversion)
        if side:
            # Entry long: price touches lower Bollinger band (mean-reversion)
            dataframe.loc[
                (
                    (dataframe["close"] <= dataframe["bb_lowerband"] * 1.003)
                    & (dataframe["rsi"] > 25)
                    & (dataframe["bb_width"] > 0.005)  # ensure not completely flat
                ),
                "enter_long",
            ] = 1

            # Entry short inside sideway: touching upper band
            dataframe.loc[
                (
                    (dataframe["close"] >= dataframe["bb_upperband"] * 0.997)
                    & (dataframe["rsi"] < 75)
                ),
                "enter_short",
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit rules likewise depend on regime; we set exit_long / exit_short columns.
        Use simple exits: opposite band, MACD cross, RSI extremes or price crossing EMA50.
        """
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        pair = metadata["pair"]
        up = self.is_uptrend(pair)
        down = self.is_downtrend(pair)
        side = self.is_sideway(pair)

        # UP: exit on MACD cross down, or price below EMA50, or RSI > 80 (overbought)
        if up:
            dataframe.loc[
                (
                    qtpylib.crossed_below(dataframe["macd"], dataframe["macdsignal"])
                    | (dataframe["close"] < dataframe["ema50"])
                    | (dataframe["rsi"] > 80)
                ),
                "exit_long",
            ] = 1

        # DOWN: cover (exit short) on MACD cross up or price above EMA50 or RSI < 30 (oversold)
        if down:
            dataframe.loc[
                (
                    qtpylib.crossed_above(dataframe["macd"], dataframe["macdsignal"])
                    | (dataframe["close"] > dataframe["ema50"])
                    | (dataframe["rsi"] < 30)
                ),
                "exit_short",
            ] = 1

        # SIDE: exit long when price at BB middle or upper band; exit short when at BB middle or lower band
        if side:
            dataframe.loc[
                (
                    (dataframe["close"] >= dataframe["bb_middleband"] * 0.998)
                    | (dataframe["rsi"] > 65)
                ),
                "exit_long",
            ] = 1

            dataframe.loc[
                (
                    (dataframe["close"] <= dataframe["bb_middleband"] * 1.002)
                    | (dataframe["rsi"] < 35)
                ),
                "exit_short",
            ] = 1

        return dataframe

    # Optional: custom stoploss (ATR based)
    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        """
        Example ATR-based dynamic stoploss: widen stoploss if ATR large.
        Freqtrade will call this if available; return None to use default stoploss.
        """
        try:
            df_1h = self.dp.get_pair_dataframe(pair, "1h")
            atr1h = ta.ATR(df_1h, timeperiod=14).iloc[-1]
            # Define a stoploss based on ATR relative to price
            atr_pct = atr1h / df_1h["close"].iloc[-1]
            # If volatility high, allow larger stoploss (but bounded)
            stop = max(self.stoploss, -0.02 - atr_pct * 5)
            return stop
        except Exception:
            return None
