# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


# Tên chiến lược: Theo dõi xu hướng tăng, mua khi giá hồi về EMA 50
class UptrendFollowStrategy(IStrategy):
    """
    Chiến lược Trend Following (Mua khi giá hồi) trong Uptrend mạnh.
    Sử dụng EMA 50/200 và ADX để lọc xu hướng, đặt lệnh Limit Order để tối ưu phí Maker.
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Không sử dụng ROI cố định, mà dựa vào Trailing Stop Loss/Take Profit
    minimal_roi = {
        "0": 10.0  # ROI cực cao để TSL (Trailing Stop Loss) làm việc thay thế
    }

    # Stoploss lỏng (để TSL và exit signal làm việc)
    stoploss = -0.99

    # ----------------------------------------------------------------------------------

    # Kích hoạt Trailing Stop
    trailing_stop = True
    # Trailing Stop sẽ bắt đầu kích hoạt sau khi đạt lãi 1.5%
    trailing_stop_positive = 0.015
    # Biên độ quay đầu 5% so với đỉnh (0.05) - nằm trong phạm vi 5-7% yêu cầu
    trailing_stop_positive_offset = 0.05
    # Chỉ kích hoạt TSL khi đạt ngưỡng lãi (trailing_stop_positive)
    trailing_only_offset_is_reached = True

    # Khung thời gian tối ưu: 1h để bắt sóng lớn
    timeframe = "1h"

    # Số lượng nến khởi động
    startup_candle_count: int = 200

    # ----------------------------------------------------------------------------------
    # Tối ưu hóa phí giao dịch: Ưu tiên Maker Order
    order_types = {
        'entry': 'limit', # Entry Mua: Limit Order (Maker Fee)
        'exit': 'limit',  # Entry Bán: Limit Order (Maker Fee)
        'stoploss': 'market', # Thoát lệnh Khẩn cấp: Market Order (Taker Fee)
        'stoploss_on_exchange': True # Đặt SL lên sàn để đảm bảo khớp lệnh ngay
    }
    order_time_in_force = {'entry': 'GTC', 'exit': 'GTC'} # Good 'Til Cancelled (Lệnh tồn tại lâu)

    # ----------------------------------------------------------------------------------
    # Hyperopt Parameters (Các thông số tối ưu hoá)

    # Ngưỡng ADX để xác định xu hướng mạnh
    adx_level = IntParameter(20, 40, default=22, space='strategy', optimize=True, load=True)
    # Ngưỡng đệm cho lệnh Limit Buy so với Hỗ trợ (tăng từ 0.998 lên 0.999 để tăng khả năng khớp)
    ema_offset = DecimalParameter(0.998, 0.9999, default=0.999, space='buy', optimize=True, load=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Tính toán các chỉ báo: EMA 20, EMA 50, EMA 200, ADX, DI+, DI-
        """

        # 1. Trend Filter (Bộ lọc Xu hướng)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        # 2. Strength of Trend (Sức mạnh Xu hướng)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Logic Entry Mua: Mua khi giá hồi về EMA 50 trong Uptrend mạnh.
        """

        # --- BỘ LỌC XU HƯỚNG CHUNG (Cần thiết cho Trend Following) ---
        uptrend_filter = (
            # 1. Giá nằm trên EMA 200 (Xu hướng lớn là tăng)
            (dataframe['close'] > dataframe['ema200']) & 
            # 2. Giá nằm trên EMA 50 (Xu hướng trung hạn là tăng)
            (dataframe['close'] > dataframe['ema50']) & 
            # 3. Sức mạnh xu hướng ADX > ngưỡng tối thiểu HOẶC phe Mua mạnh hơn phe Bán (DI+ > DI-)
            (
                (dataframe['adx'] > self.adx_level.value) |  # ADX > 22-25
                (dataframe['plus_di'] > dataframe['minus_di'])
            )
        )

        # --- TÍN HIỆU 1: HỒI SÂU VỀ EMA 50 (Cơ hội R:R tốt nhất) ---
        retrace_ema50 = (
            # Giá nến trước cao hơn EMA 50
            (dataframe['open'].shift(1) > dataframe['ema50'].shift(1)) & 
            # Giá đóng cửa hiện tại chạm hoặc thủng EMA 50 (hồi đủ sâu)
            (dataframe['close'] < dataframe['ema50']) &
            # Giá mở cửa hiện tại cao hơn giá đóng cửa (dấu hiệu bật lại)
            (dataframe['open'] > dataframe['close']) 
        )

        # --- TÍN HIỆU 2: HỒI NÔNG VỀ EMA 20 (Thường xảy ra trong Bull mạnh) ---
        retrace_ema20 = (
            # Giá nằm trên EMA 50 (Xu hướng mạnh)
            (dataframe['close'] > dataframe['ema50']) & 
            # Giá nến trước cao hơn EMA 20
            (dataframe['open'].shift(1) > dataframe['ema20'].shift(1)) & 
            # Giá đóng cửa hiện tại chạm hoặc thủng EMA 20
            (dataframe['close'] < dataframe['ema20']) &
            # Giá mở cửa hiện tại cao hơn giá đóng cửa (dấu hiệu bật lại)
            (dataframe['open'] > dataframe['close'])
        )

        # Lệnh Mua sẽ được kích hoạt nếu thỏa mãn bộ lọc chung VÀ (Tín hiệu 1 HOẶC Tín hiệu 2)
        dataframe.loc[
            uptrend_filter & (retrace_ema50 | retrace_ema20),
            'enter_long',
        ] = 1

        # Gán giá vào lệnh (Enter Rate) để tối ưu Maker Fee
        dataframe.loc[
            (dataframe['enter_long'] == 1),
            'enter_tag',
        ] = 'Retrace_EMA_Signal'

        # Giá Limit Order: Đặt lệnh Limit Buy ngay dưới mức Hỗ trợ (EMA 50 hoặc EMA 20) một chút
        # Tùy thuộc vào tín hiệu nào được kích hoạt, ta đặt lệnh sát Hỗ trợ đó.
        dataframe.loc[
            (dataframe['enter_long'] == 1) & retrace_ema50,
            'enter_rate',
        ] = dataframe['ema50'] * self.ema_offset.value

        dataframe.loc[
            (dataframe['enter_long'] == 1) & retrace_ema20,
            'enter_rate',
        ] = dataframe['ema20'] * self.ema_offset.value
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Logic Exit Bán: Không cần tín hiệu exit cố định vì sử dụng Trailing Stop Loss (TSL).
        Tuy nhiên, để đảm bảo khớp lệnh Maker, ta có thể đặt lệnh Limit Sell trên giá đỉnh cục bộ.
        """

        # Nếu không sử dụng TSL, có thể dùng logic bán khi giá chạm dải trên Bollinger.
        # Nhưng ở đây TSL làm nhiệm vụ chính, nên logic này có thể để trống, hoặc dùng cho các lệnh lỗi
        # dataframe.loc[
        #     (dataframe['close'] > dataframe['bb_upperband']),
        #     'exit_long',
        # ] = 1

        # Với TSL/TTP, ta chủ yếu dựa vào các tham số đã cấu hình.
        # Tuy nhiên, để tối ưu phí Exit (Limit Order/Maker), ta có thể đặt một lệnh chốt lời tiềm năng.

        if 'exit_long' in dataframe.columns and not dataframe['exit_long'].empty:
             # Đặt lệnh Limit Sell (Maker) trên giá đóng cửa một chút, chờ khớp
             # TSL sẽ tự động thay thế lệnh này nếu cần cắt lỗ/chốt lời khẩn cấp.
            dataframe.loc[
                (dataframe['exit_long'] == 1),
                'exit_rate',
            ] = dataframe['close'] * 1.002 # Ví dụ: đặt chốt lãi 0.2% trên giá đóng cửa

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        initial_rate: float, current_profit: float) -> Union[float, None]:
        """
        Logic Stoploss Tùy chỉnh: Đặt SL dưới EMA 200 hoặc dưới đáy gần nhất.
        Trong Freqtrade, cách đơn giản nhất là đặt Stoploss dưới ngưỡng EMA 200,
        và Freqtrade sẽ tự động dịch chuyển SL theo giá trị này nếu giá tăng.
        """

        # Lấy dữ liệu 1h của cặp tiền
        dataframe, _ = self.dp.get_pair_dataframe(pair, self.timeframe)

        # Lấy giá trị EMA 200 tại thời điểm hiện tại (nến cuối cùng)
        ema200_value = dataframe['ema200'].iloc[-1]

        # Tính toán mức lỗ tuyệt đối cần đặt:
        # Nếu SL = -0.5, có nghĩa là 50% dưới giá vào (initial_rate * 0.5)
        # Chúng ta muốn SL ở mức giá EMA 200.

        # Ví dụ: Giá vào 1000, EMA 200 là 950. Tức là SL 5%
        # Công thức: (SL_Price / Initial_Rate) - 1
        sl_percent = (ema200_value / trade.open_rate) - 1

        # Đảm bảo Stoploss nằm dưới mức giá vào (sl_percent phải là số âm)
        # Nếu EMA 200 tăng vượt giá vào, SL sẽ bị chặn ở mức 0 (hòa vốn).
        return max(sl_percent, -0.99) # Cắt lỗ sâu nhất -99% nếu EMA 200 quá xa.