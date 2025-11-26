from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from technical import qtpylib


class SidewayGridStrategy(IStrategy):
    """
    Sideway-only trading strategy.
    - Pause trading in uptrend/downtrend.
    - Trade only when market is truly sideway using 1h trend filters.
    - Grid-friendly entry (BB lower band) + trailing stop for exit.
    """

    INTERFACE_VERSION = 3
    can_short = False
    timeframe = "5m"
    informative_timeframe = "1h"

    # ROI & SL
    minimal_roi = {"0": 0.02}
    stoploss = -0.04

    # Trailing Stop
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    # Candles needed
    startup_candle_count = 300

    # Settings
    bb_window = 20
    bb_std = 2

    rsi_buy = 45
    rsi_sell = 60

    # Sideway filters
    adx_limit = 18       # ADX < 18 nghĩa là thị trường yếu, không có trend
    bb_width_limit = 0.015
    atr_rel_limit = 0.015

    def informative_pairs(self):
        """Use 1h timeframe for detecting sideway/uptrend/downtrend."""
        pairs = [(pair, self.informative_timeframe) for pair in self.dp.current_whitelist()]
        return pairs

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:

        # ------------------
        # 1) 5m indicators
        # ------------------
        df['rsi'] = ta.RSI(df, timeperiod=14)

        boll = qtpylib.bollinger_bands(
            qtpylib.typical_price(df),
            window=self.bb_window,
            stds=self.bb_std
        )
        df['bb_lowerband'] = boll['lower']
        df['bb_middleband'] = boll['mid']
        df['bb_upperband'] = boll['upper']
        df['bb_width'] = (df['bb_upperband'] - df['bb_lowerband']) / df['bb_middleband']

        df['atr'] = ta.ATR(df, timeperiod=14)
        df['ema50'] = ta.EMA(df, timeperiod=50)
        df['ema200'] = ta.EMA(df, timeperiod=200)

        # ------------------
        # 2) 1h informative indicators (trend detection)
        # ------------------
        informative = self.dp.get_pair_dataframe(
            pair=metadata['pair'],
            timeframe=self.informative_timeframe
        )

        informative['ema50'] = ta.EMA(informative, timeperiod=50)
        informative['ema200'] = ta.EMA(informative, timeperiod=200)
        informative['adx'] = ta.ADX(informative)
        informative['atr'] = ta.ATR(informative)

        boll_h = qtpylib.bollinger_bands(
            qtpylib.typical_price(informative),
            window=20,
            stds=2
        )
        informative['bb_width'] = (boll_h['upper'] - boll_h['lower']) / boll_h['mid']

        # Merge into 5m
        df = merge_informative_pair(df, informative, self.timeframe, self.informative_timeframe, ffill=True)

        return df

    # -----------------------------
    # SIDEWAY DETECTION (1h filters)
    # -----------------------------
    def is_sideway(self, df: DataFrame) -> DataFrame:
        """
        Điều kiện sideway trên khung 1h:
        - EMA50 gần EMA200 (không phân kỳ mạnh)
        - ADX < 18 (trend yếu)
        - BB width hẹp
        - ATR thấp => thị trường bình lặng
        """

        df['sideway'] = (
            (abs(df['ema50_1h'] - df['ema200_1h']) / df['ema200_1h'] < 0.03) &  # EMA gần nhau
            (df['adx_1h'] < self.adx_limit) &                                  # Trend yếu
            (df['bb_width_1h'] < self.bb_width_limit) &                        # Dao động hẹp
            ((df['atr_1h'] / df['close_1h']) < self.atr_rel_limit)             # ATR thấp
        )

        return df

    # --------------------------------
    # ENTRY
    # --------------------------------
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df = self.is_sideway(df)  # Gọi hàm sideway

        # Thêm volume 20 cây trung bình
        df['volume_mean'] = df['volume'].rolling(20).mean()

        # Chỉ được buy khi thị trường đang sideway
        df.loc[
            (
                (df['sideway']) &
                (df['close'] <= df['bb_lowerband'] * 1.003) & # Mua khi chạm BB lower
                (df['rsi'] > 30) & (df['rsi'] < self.rsi_buy) & # Mua khi oversold
                (df['bb_width'] > 0.005) & # BB width phải đủ rộng một chút. Tránh trade khi quá phẳng (không có biên độ lãi)
                ((df['atr'] / df['close']) > 0.001) & # ATR phải đủ lớn. Tránh trade khi quá phẳng (không có biên độ lãi)
                (df['volume'] > df['volume_mean'] * 3) & # Volume bùng nổ. Loại fake dip
                (df['close'] > df['open']) # Nến xanh xác nhận. Loại pump giả, giảm stoploss
            ),
            'enter_long'
        ] = 1

        return df

    # --------------------------------
    # EXIT
    # --------------------------------
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        # Exit khi có lãi nhẹ hoặc RSI cao
        df.loc[
            (
                (df['close'] >= df['bb_middleband'] * 1.002) &
                (df['rsi'] > self.rsi_sell)
            ),
            'exit_long'
        ] = 1

        # Exit khi chạm Upper BB
        df.loc[
            (df['close'] >= df['bb_upperband'] * 0.998),
            'exit_long'
        ] = 1

        return df
