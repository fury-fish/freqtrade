# freqtrade strategies pack: 3 strategies for Sideway / Downtrend / Uptrend markets
# Drop this file into user_data/strategies/ and import classes individually (one class per strategy file is fine).
# Each class is self-contained and ready to run with Freqtrade v2023+ (interface v3).

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from technical import qtpylib

# ----------------------------
# 1) Sideway strategy (Grid-friendly)
# ----------------------------
class SidewayGridStrategy(IStrategy):
    """Grid-friendly strategy for sideway markets.
    Logic summary:
      - Use 1h EMA200 + BB width and ATR to confirm sideway.
      - Buy near lower Bollinger Band when RSI not extreme and volume ok.
      - Sell near upper Bollinger Band or when target profit reached.
    Notes: This is a *signal-based* grid-friendly approach. For a true order-grid
    engine consider pairing this with a separate order-manager that places multiple
    limit levels. This strategy gives conservative entry/exit signals suitable
    for grid-like behavior on 5m timeframe.
    """

    INTERFACE_VERSION = 3
    can_short = False
    timeframe = "5m"
    minimal_roi = {"0": 0.015}
    stoploss = -0.12
    startup_candle_count = 300

    # Hyperoptable params (examples)
    bb_window = 20
    bb_std = 2
    rsi_buy = 45
    rsi_sell = 60
    atr_threshold = 0.002  # relative to price

    def informative_pairs(self):
        # Use 1h for trend/sideway detection
        return [(pair, "1h") for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Standard indicators
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        boll = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=self.bb_window, stds=self.bb_std)
        dataframe['bb_lowerband'] = boll['lower']
        dataframe['bb_middleband'] = boll['mid']
        dataframe['bb_upperband'] = boll['upper']
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # informative 1h EMA200 & ATR -> merge via merge_informative_pair when used in Freqtrade
        # We'll compute local EMA50/200 on 5m as well for quick guards
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Entry: price near lower BB, moderate RSI, BB width not tiny (has movement), ATR not too large
        dataframe.loc[
            (
                (dataframe['close'] <= dataframe['bb_lowerband'] * 1.003) &
                (dataframe['rsi'] > 30) & (dataframe['rsi'] < self.rsi_buy) &
                (dataframe['bb_width'] > 0.01) &
                (dataframe['atr'] / dataframe['close'] > self.atr_threshold)
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit: price near upper BB or RSI above threshold
        dataframe.loc[
            (
                (dataframe['close'] >= dataframe['bb_middleband'] * 1.002) &
                (dataframe['rsi'] > self.rsi_sell)
            ),
            'exit_long'
        ] = 1

        # Also exit if price touches upper band
        dataframe.loc[(dataframe['close'] >= dataframe['bb_upperband'] * 0.998), 'exit_long'] = 1

        return dataframe
