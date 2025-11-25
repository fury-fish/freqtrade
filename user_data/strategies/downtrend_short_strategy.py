# freqtrade strategies pack: 3 strategies for Sideway / Downtrend / Uptrend markets
# Drop this file into user_data/strategies/ and import classes individually (one class per strategy file is fine).
# Each class is self-contained and ready to run with Freqtrade v2023+ (interface v3).

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from technical import qtpylib

# ----------------------------
# 2) Downtrend strategy (Short trend-following)
# ----------------------------
class DowntrendShortStrategy(IStrategy):
    """Short-focused trend-following strategy for downtrend markets.
    Requires futures/live environment with short enabled.
    Logic summary:
      - Use 1h EMA50/EMA200 (informative) to confirm downtrend
      - Use ADX to ensure trend strength
      - Enter short on pullback (price retraces to EMA20/50) with bearish confirmation
      - Exit on fixed TP or when trend weakens
    """

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '5m'
    minimal_roi = {"0": 0.005}
    stoploss = -0.06
    startup_candle_count = 300

    # Parameters
    adx_threshold = 20
    pullback_ema = 20

    def informative_pairs(self):
        return [(pair, '1h') for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Basic indicators
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        return dataframe

    def is_downtrend_informative(self, pair: str) -> bool:
        # helper using informative 1h dataframe
        inf_1h = self.dp.get_pair_dataframe(pair, '1h')
        ema50_1h = ta.EMA(inf_1h, timeperiod=50).iloc[-1]
        ema200_1h = ta.EMA(inf_1h, timeperiod=200).iloc[-1]
        price_1h = inf_1h['close'].iloc[-1]
        # define downtrend if ema50 < ema200 and price below ema200 by >1%
        return (ema50_1h < ema200_1h) and (price_1h < ema200_1h * 0.99)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        downtrend_ok = False
        try:
            downtrend_ok = self.is_downtrend_informative(pair)
        except Exception:
            # if informative not available, fallback to local ema check
            downtrend_ok = (dataframe['ema50'].iloc[-1] < dataframe['ema200'].iloc[-1])

        # Entry short: in confirmed downtrend, on pullback to ema20 or when macd signal turns down
        dataframe.loc[
            (
                downtrend_ok &
                (dataframe['close'] >= dataframe['ema20']) &
                (dataframe['macd'] < dataframe['macdsignal']) &
                (dataframe['adx'] > self.adx_threshold) &
                (dataframe['rsi'] < 80)
            ),
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit short: MACD crossing up, or RSI below 30 (oversold cover), or price above EMA50
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])) |
                (dataframe['rsi'] < 30) |
                (dataframe['close'] > dataframe['ema50'])
            ),
            'exit_short'
        ] = 1

        return dataframe
