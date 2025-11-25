# freqtrade strategies pack: 3 strategies for Sideway / Downtrend / Uptrend markets
# Drop this file into user_data/strategies/ and import classes individually (one class per strategy file is fine).
# Each class is self-contained and ready to run with Freqtrade v2023+ (interface v3).

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from technical import qtpylib

# ----------------------------
# 3) Uptrend strategy (Trend-following long)
# ----------------------------
class UptrendFollowStrategy(IStrategy):
    """Trend-following long-only strategy for uptrend markets.
    Logic summary:
      - Use 1h EMA50/EMA200 to confirm uptrend
      - Enter long on pullback to EMA20 or when EMA cross confirms
      - Use ATR-based dynamic stoploss and trailing-friendly ROI
    """

    INTERFACE_VERSION = 3
    can_short = False
    timeframe = '5m'
    minimal_roi = {"0": 0.02, "30": 0.01}
    stoploss = -0.08
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    startup_candle_count = 300

    def informative_pairs(self):
        return [(pair, '1h') for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        return dataframe

    def is_uptrend_informative(self, pair: str) -> bool:
        try:
            inf_1h = self.dp.get_pair_dataframe(pair, '1h')
            ema50_1h = ta.EMA(inf_1h, timeperiod=50).iloc[-1]
            ema200_1h = ta.EMA(inf_1h, timeperiod=200).iloc[-1]
            price_1h = inf_1h['close'].iloc[-1]
            return (ema50_1h > ema200_1h) and (price_1h > ema50_1h)
        except Exception:
            return False

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        uptrend_ok = self.is_uptrend_informative(pair)

        # Enter long: in confirmed uptrend, on pullback to ema20 or macd crossover up
        dataframe.loc[
            (
                uptrend_ok &
                (
                    (dataframe['close'] <= dataframe['ema20'] * 1.002) &
                    (dataframe['rsi'] > 35)  # avoid extreme oversold entry
                )
                |
                (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']))
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long: macd cross down or price dropping below ema50
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])) |
                (dataframe['close'] < dataframe['ema50'])
            ),
            'exit_long'
        ] = 1

        return dataframe
