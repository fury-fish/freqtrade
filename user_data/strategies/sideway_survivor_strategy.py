from freqtrade.strategy import IStrategy
import talib.abstract as ta
import pandas as pd

class SidewaySurvivor(IStrategy):
    timeframe = '5m'
    minimal_roi = {"0": 0.03}           # Lãi 3% là thoát luôn
    stoploss = -0.02                    # Cắt lỗ nhanh -2%
    trailing_stop = True
    trailing_stop_positive = 0.01       # Lãi 1% thì bật trailing
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_sma'] = dataframe['volume'].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < 32) &
                (dataframe['volume'] > dataframe['volume_sma'] * 2) &  # Volume bùng nổ
                (dataframe['ema20'] > dataframe['ema50']) &           # Nhẹ nhàng uptrend
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (dataframe['rsi'] > 68),
            'exit_long'] = 1
        return dataframe