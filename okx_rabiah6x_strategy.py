"""
Rabiah6X - UT Bot + 1:1 Scalp Targets + DEMA + FVG
Python implementation for OKX automated trading with logging system.
Fully replicates the TradingView Pine Script strategy.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import time

import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============ LOGGING CONFIGURATION ============
def setup_logging(log_file: str = "trading_bot.log", level: int = logging.INFO) -> logging.Logger:
    """
    Setup comprehensive logging system with file and console handlers.
    """
    logger = logging.getLogger("Rabiah6X_Bot")
    logger.setLevel(level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - all logs
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler - info and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()


# ============ ENUMS AND DATA CLASSES ============
class PositionSide(Enum):
    NONE = 0
    LONG = 1
    SHORT = -1


@dataclass
class TradeConfig:
    """Configuration for trading parameters."""
    # UT Bot Settings
    key_value: float = 2.0
    atr_period: int = 10
    htf_filter_on: bool = False
    htf_timeframe: str = "60m"
    vol_filter_on: bool = False
    vol_len: int = 20
    
    # DEMA Settings
    dema_len: int = 21
    
    # Early Exit Settings
    rr_multiplier: float = 1.0
    use_dema_exit: bool = True
    use_rsi_exit: bool = True
    rsi_len: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    
    # FVG & Momentum Settings
    use_fvg_filter: bool = True
    use_adx_filter: bool = True
    adx_len: int = 14
    adx_thresh: int = 20
    body_ratio: float = 0.6
    
    # Risk Management
    initial_capital: float = 2000.0
    position_size_pct: float = 100.0  # percent of equity
    
    # OKX Specific
    symbol: str = "BTC/USDT"
    timeframe: str = "5m"
    leverage: int = 1


@dataclass
class TradeState:
    """Current state of the trading bot."""
    position_side: PositionSide = PositionSide.NONE
    entry_price: float = 0.0
    position_size: float = 0.0
    trailing_stop: float = 0.0
    target_price: float = 0.0
    last_signal_time: Optional[datetime] = None
    
    # Performance tracking
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    
    # Signal history
    signal_history: list = field(default_factory=list)


# ============ TECHNICAL INDICATORS ============
class TechnicalIndicators:
    """Calculate technical indicators matching Pine Script logic."""
    
    @staticmethod
    def ema(series: pd.Series, length: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=length, adjust=False).mean()
    
    @staticmethod
    def dema(series: pd.Series, length: int) -> pd.Series:
        """Calculate Double Exponential Moving Average (DEMA)."""
        ema1 = TechnicalIndicators.ema(series, length)
        ema2 = TechnicalIndicators.ema(ema1, length)
        return 2 * ema1 - ema2
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def rsi(close: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average Directional Index (ADX)."""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = TechnicalIndicators.atr(high, low, close, 1)
        
        plus_di = 100 * (plus_dm.rolling(window=period).sum() / 
                         tr.rolling(window=period).sum())
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / 
                          tr.rolling(window=period).sum())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def detect_fvg(high: pd.Series, low: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Detect Fair Value Gaps (FVG).
        Returns bullish and bearish FVG boolean series.
        """
        # Bullish FVG: current low > 2 bars ago high
        bullish_fvg = low > high.shift(2)
        
        # Bearish FVG: current high < 2 bars ago low
        bearish_fvg = high < low.shift(2)
        
        return bullish_fvg, bearish_fvg
    
    @staticmethod
    def detect_recent_fvg(bullish_fvg: pd.Series, bearish_fvg: pd.Series, lookback: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Check if FVG occurred within recent bars."""
        recent_bull_fvg = bullish_fvg.rolling(window=lookback, min_periods=1).max().astype(bool)
        recent_bear_fvg = bearish_fvg.rolling(window=lookback, min_periods=1).max().astype(bool)
        return recent_bull_fvg, recent_bear_fvg
    
    @staticmethod
    def detect_expansion_bar(open_: pd.Series, high: pd.Series, low: pd.Series, 
                             close: pd.Series, atr: pd.Series, body_ratio: float) -> pd.Series:
        """
        Detect expansion bars (strong momentum candles).
        Conditions:
        - Body size / candle range >= body_ratio
        - Body size > ATR * 0.7
        """
        body_size = abs(close - open_)
        candle_range = high - low
        
        condition1 = (candle_range > 0) & ((body_size / candle_range) >= body_ratio)
        condition2 = body_size > (atr * 0.7)
        
        return condition1 & condition2


# ============ UT BOT CALCULATOR ============
class UTBotCalculator:
    """Calculate UT Bot trailing stop and position signals."""
    
    def __init__(self, key_value: float, atr_period: int):
        self.key_value = key_value
        self.atr_period = atr_period
    
    def calculate(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate UT Bot trailing stop and position.
        Returns: (trailing_stop, position, xATR)
        """
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Calculate ATR
        xATR = TechnicalIndicators.atr(high, low, close, self.atr_period)
        nLoss = self.key_value * xATR
        
        # Initialize arrays
        n = len(df)
        xATRTrailingStop = np.zeros(n)
        pos = np.zeros(n, dtype=int)
        
        # Calculate trailing stop iteratively (matching Pine Script logic)
        prev_stop = 0.0
        prev_pos = 0
        
        for i in range(n):
            src = close.iloc[i]
            src_prev = close.iloc[i-1] if i > 0 else src
            
            if i == 0:
                xATRTrailingStop[i] = src - nLoss.iloc[i]
                pos[i] = 0
            else:
                # Calculate trailing stop
                if src_prev > prev_stop and src > prev_stop:
                    xATRTrailingStop[i] = max(prev_stop, src - nLoss.iloc[i])
                elif src_prev < prev_stop and src < prev_stop:
                    xATRTrailingStop[i] = min(prev_stop, src + nLoss.iloc[i])
                elif src > prev_stop:
                    xATRTrailingStop[i] = src - nLoss.iloc[i]
                else:
                    xATRTrailingStop[i] = src + nLoss.iloc[i]
                
                # Determine position
                current_stop = xATRTrailingStop[i]
                if src_prev < prev_stop and src > prev_stop:
                    pos[i] = 1
                elif src_prev > prev_stop and src < prev_stop:
                    pos[i] = -1
                else:
                    pos[i] = prev_pos
            
            prev_stop = xATRTrailingStop[i]
            prev_pos = pos[i]
        
        return (
            pd.Series(xATRTrailingStop, index=df.index),
            pd.Series(pos, index=df.index),
            xATR
        )


# ============ SIGNAL GENERATOR ============
class SignalGenerator:
    """Generate trading signals based on UT Bot strategy."""
    
    def __init__(self, config: TradeConfig):
        self.config = config
        self.ut_bot = UTBotCalculator(config.key_value, config.atr_period)
        self.indicators = TechnicalIndicators()
    
    def calculate_all_indicators(self, df: pd.DataFrame, 
                                  htf_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate all indicators needed for signal generation."""
        df = df.copy()
        
        # DEMA
        df['dema'] = self.indicators.dema(df['close'], self.config.dema_len)
        
        # RSI
        df['rsi'] = self.indicators.rsi(df['close'], self.config.rsi_len)
        
        # ADX
        df['adx'] = self.indicators.adx(df['high'], df['low'], df['close'], self.config.adx_len)
        
        # FVG Detection
        bullish_fvg, bearish_fvg = self.indicators.detect_fvg(df['high'], df['low'])
        df['bullish_fvg'] = bullish_fvg
        df['bearish_fvg'] = bearish_fvg
        
        recent_bull, recent_bear = self.indicators.detect_recent_fvg(bullish_fvg, bearish_fvg)
        df['recent_bull_fvg'] = recent_bull
        df['recent_bear_fvg'] = recent_bear
        
        # ATR and Expansion Bar
        df['atr'] = self.indicators.atr(df['high'], df['low'], df['close'], self.config.atr_period)
        df['expansion_bar'] = self.indicators.detect_expansion_bar(
            df['open'], df['high'], df['low'], df['close'], 
            df['atr'], self.config.body_ratio
        )
        
        # UT Bot
        trailing_stop, position, xATR = self.ut_bot.calculate(df)
        df['trailing_stop'] = trailing_stop
        df['ut_position'] = position
        
        # Volume Filter
        if self.config.vol_filter_on:
            df['vol_ma'] = df['volume'].rolling(window=self.config.vol_len).mean()
            df['vol_ok'] = df['volume'] > df['vol_ma']
        else:
            df['vol_ok'] = True
        
        # Higher Timeframe Filter
        if self.config.htf_filter_on and htf_df is not None:
            htf_sma = htf_df['close'].rolling(window=50).mean().iloc[-1]
            htf_close = htf_df['close'].iloc[-1]
            df['htf_trend_up'] = htf_close > htf_sma
        else:
            df['htf_trend_up'] = True
        
        # Strong Trend Check
        if self.config.use_adx_filter:
            df['is_strong_trend'] = df['adx'] >= self.config.adx_thresh
        else:
            df['is_strong_trend'] = True
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals for the latest bar.
        Returns dictionary with signal information.
        """
        if len(df) < 5:
            return {'signal': 'NONE', 'reason': 'Insufficient data'}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Base signals
        base_buy = (prev['close'] < prev['trailing_stop']) and \
                   (latest['close'] > latest['trailing_stop']) and \
                   latest['vol_ok'] and latest['htf_trend_up']
        
        base_sell = (prev['close'] > prev['trailing_stop']) and \
                    (latest['close'] < latest['trailing_stop']) and \
                    latest['vol_ok'] and (not latest['htf_trend_up'] if self.config.htf_filter_on else True)
        
        # Strong signals with FVG confluence
        is_strong_buy = (base_buy and 
                        latest['recent_bull_fvg'] and 
                        latest['is_strong_trend'] and 
                        latest['expansion_bar'])
        
        is_standard_buy = base_buy and not is_strong_buy
        
        is_strong_sell = (base_sell and 
                         latest['recent_bear_fvg'] and 
                         latest['is_strong_trend'] and 
                         latest['expansion_bar'])
        
        is_standard_sell = base_sell and not is_strong_sell
        
        # Early exit signals
        dema_exit_long = (self.config.use_dema_exit and 
                         latest['close'] < latest['dema'])
        
        dema_exit_short = (self.config.use_dema_exit and 
                          latest['close'] > latest['dema'])
        
        rsi_exit_long = (self.config.use_rsi_exit and 
                        latest['rsi'] < self.config.rsi_overbought)
        
        rsi_exit_short = (self.config.use_rsi_exit and 
                         latest['rsi'] > self.config.rsi_oversold)
        
        # Determine primary signal
        if is_strong_buy:
            signal = 'STRONG_BUY'
            reason = 'UT Bot crossover + FVG + Strong Trend + Expansion'
        elif is_standard_buy:
            signal = 'BUY'
            reason = 'UT Bot crossover'
        elif is_strong_sell:
            signal = 'STRONG_SELL'
            reason = 'UT Bot crossunder + FVG + Strong Trend + Expansion'
        elif is_standard_sell:
            signal = 'SELL'
            reason = 'UT Bot crossunder'
        elif dema_exit_long or rsi_exit_long:
            signal = 'EXIT_LONG'
            reason = 'Early exit (DEMA/RSI)'
        elif dema_exit_short or rsi_exit_short:
            signal = 'EXIT_SHORT'
            reason = 'Early exit (DEMA/RSI)'
        else:
            signal = 'NONE'
            reason = 'No clear signal'
        
        # Calculate target prices
        target_price_long = None
        target_price_short = None
        
        if base_buy:
            risk = latest['close'] - latest['trailing_stop']
            target_price_long = latest['close'] + (risk * self.config.rr_multiplier)
        
        if base_sell:
            risk = latest['trailing_stop'] - latest['close']
            target_price_short = latest['close'] - (risk * self.config.rr_multiplier)
        
        return {
            'signal': signal,
            'reason': reason,
            'timestamp': latest.name,
            'price': latest['close'],
            'trailing_stop': latest['trailing_stop'],
            'target_price_long': target_price_long,
            'target_price_short': target_price_short,
            'dema': latest['dema'],
            'rsi': latest['rsi'],
            'adx': latest['adx'],
            'is_strong_buy': is_strong_buy,
            'is_standard_buy': is_standard_buy,
            'is_strong_sell': is_strong_sell,
            'is_standard_sell': is_standard_sell,
            'dema_exit_long': dema_exit_long,
            'dema_exit_short': dema_exit_short,
            'rsi_exit_long': rsi_exit_long,
            'rsi_exit_short': rsi_exit_short
        }


# ============ OKX TRADING ENGINE ============
class OKXTradingEngine:
    """Handle OKX exchange operations."""
    
    def __init__(self, config: TradeConfig):
        self.config = config
        self.exchange = None
        self.is_connected = False
        
        # Credentials from environment
        self.api_key = os.getenv('OKX_API_KEY')
        self.api_secret = os.getenv('OKX_API_SECRET')
        self.passphrase = os.getenv('OKX_PASSPHRASE')
        self.sandbox_mode = os.getenv('OKX_SANDBOX', 'true').lower() == 'true'
        
        if not all([self.api_key, self.api_secret, self.passphrase]):
            logger.warning("OKX credentials not found in environment variables. Running in simulation mode.")
            self.simulation_mode = True
        else:
            self.simulation_mode = False
    
    async def connect(self) -> bool:
        """Connect to OKX exchange."""
        try:
            if self.simulation_mode:
                logger.info("Running in SIMULATION MODE - no real trades will be executed")
                self.is_connected = True
                return True
            
            # OKX требует использования sandboxMode только для демо-ключей
            # Для реальных ключей используйте production режим
            self.exchange = ccxt.okx({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.passphrase,
                'enableRateLimit': True,
                'sandboxMode': self.sandbox_mode,  # True для демо-ключей, False для реальных
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'warnOnFetchOpenOrdersWithoutSymbol': False
                }
            })
            
            # Load markets
            await self.exchange.load_markets()
            
            # Set leverage
            if self.config.leverage > 1:
                await self.exchange.set_leverage(self.config.leverage, self.config.symbol)
            
            self.is_connected = True
            logger.info(f"Connected to OKX. Symbol: {self.config.symbol}, Leverage: {self.config.leverage}x, Sandbox: {self.sandbox_mode}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to connect to OKX: {error_msg}")
            
            # Подсказка для пользователя
            if "APIKey does not match" in error_msg or "50101" in error_msg:
                logger.error(">>> ОШИБКА: API-ключи не соответствуют среде (Sandbox/Production)")
                logger.error(">>> Решение:")
                logger.error(">>>   - Если используете ДЕМО-ключи: установите OKX_SANDBOX=true в .env")
                logger.error(">>>   - Если используете РЕАЛЬНЫЕ ключи: установите OKX_SANDBOX=false в .env")
                logger.error(">>>   - Демо-ключи создаются в разделе 'Demo Trading' на okx.com")
                logger.error(">>>   - Реальные ключи создаются в разделе 'API Management' на okx.com")
            
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from exchange."""
        if self.exchange:
            await self.exchange.close()
            self.is_connected = False
            logger.info("Disconnected from OKX")
    
    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from exchange."""
        try:
            if self.simulation_mode:
                # Return mock data for testing
                logger.warning(f"Simulation mode: returning mock data for {symbol}")
                return self._generate_mock_data(limit)
            
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.debug(f"Fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data: {str(e)}")
            return pd.DataFrame()
    
    def _generate_mock_data(self, limit: int) -> pd.DataFrame:
        """Generate mock OHLCV data for simulation."""
        np.random.seed(42)
        timestamps = pd.date_range(end=datetime.now(timezone.utc), periods=limit, freq='5min')
        
        base_price = 50000
        prices = base_price + np.cumsum(np.random.randn(limit) * 100)
        
        df = pd.DataFrame({
            'open': prices + np.random.randn(limit) * 10,
            'high': prices + np.abs(np.random.randn(limit) * 20),
            'low': prices - np.abs(np.random.randn(limit) * 20),
            'close': prices,
            'volume': np.random.randint(100, 1000, limit)
        }, index=timestamps)
        
        return df
    
    async def get_position(self) -> Dict[str, Any]:
        """Get current position."""
        try:
            if self.simulation_mode:
                return {'contracts': 0, 'side': 'none', 'entryPrice': 0}
            
            positions = await self.exchange.fetch_positions([self.config.symbol])
            for pos in positions:
                if pos['contracts'] and pos['contracts'] > 0:
                    return {
                        'contracts': pos['contracts'],
                        'side': pos['side'],
                        'entryPrice': pos['entryPrice']
                    }
            
            return {'contracts': 0, 'side': 'none', 'entryPrice': 0}
            
        except Exception as e:
            logger.error(f"Failed to fetch position: {str(e)}")
            return {'contracts': 0, 'side': 'none', 'entryPrice': 0}
    
    async def place_market_order(self, side: str, size: float) -> Optional[Dict]:
        """Place a market order."""
        try:
            if self.simulation_mode:
                logger.info(f"[SIMULATION] Market {side.upper()} order: {size} contracts")
                return {'id': 'sim_' + str(int(time.time())), 'side': side, 'amount': size}
            
            order = await self.exchange.create_market_order(
                self.config.symbol,
                side,
                size
            )
            
            logger.info(f"Placed market {side.upper()} order: {size} contracts. Order ID: {order['id']}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to place market order: {str(e)}")
            return None
    
    async def place_limit_order(self, side: str, size: float, price: float) -> Optional[Dict]:
        """Place a limit order."""
        try:
            if self.simulation_mode:
                logger.info(f"[SIMULATION] Limit {side.upper()} order: {size} contracts @ {price}")
                return {'id': 'sim_' + str(int(time.time())), 'side': side, 'amount': size, 'price': price}
            
            order = await self.exchange.create_limit_order(
                self.config.symbol,
                side,
                size,
                price
            )
            
            logger.info(f"Placed limit {side.upper()} order: {size} contracts @ {price}. Order ID: {order['id']}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to place limit order: {str(e)}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            if self.simulation_mode:
                logger.info(f"[SIMULATION] Cancelled order: {order_id}")
                return True
            
            await self.exchange.cancel_order(order_id, self.config.symbol)
            logger.info(f"Cancelled order: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {str(e)}")
            return False
    
    async def close_position(self) -> bool:
        """Close current position."""
        try:
            position = await self.get_position()
            
            if position['contracts'] == 0:
                logger.debug("No open position to close")
                return True
            
            side = 'sell' if position['side'] == 'long' else 'buy'
            size = abs(position['contracts'])
            
            return await self.place_market_order(side, size) is not None
            
        except Exception as e:
            logger.error(f"Failed to close position: {str(e)}")
            return False


# ============ MAIN TRADING BOT ============
class Rabiah6XBot:
    """Main trading bot implementing the Rabiah6X strategy."""
    
    def __init__(self, config: TradeConfig):
        self.config = config
        self.state = TradeState()
        self.signal_generator = SignalGenerator(config)
        self.trading_engine = OKXTradingEngine(config)
        
        # Order IDs for tracking
        self.take_profit_order_id: Optional[str] = None
        self.stop_loss_order_id: Optional[str] = None
        
        logger.info(f"Rabiah6X Bot initialized with config: {config.symbol}, TF: {config.timeframe}")
    
    async def start(self):
        """Start the trading bot."""
        logger.info("Starting Rabiah6X trading bot...")
        
        # Connect to exchange
        if not await self.trading_engine.connect():
            logger.error("Failed to connect to exchange. Exiting.")
            return
        
        try:
            while True:
                await self._trading_cycle()
                await asyncio.sleep(60)  # Wait 1 minute before next cycle
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {str(e)}")
        finally:
            await self.trading_engine.disconnect()
            self._log_final_stats()
    
    async def _trading_cycle(self):
        """Execute one trading cycle."""
        try:
            # Fetch market data
            df = await self.trading_engine.get_ohlcv(
                self.config.symbol, 
                self.config.timeframe, 
                limit=100
            )
            
            if df.empty:
                logger.warning("No market data available")
                return
            
            # Fetch HTF data if filter enabled
            htf_df = None
            if self.config.htf_filter_on:
                htf_df = await self.trading_engine.get_ohlcv(
                    self.config.symbol,
                    self.config.htf_timeframe,
                    limit=100
                )
            
            # Calculate indicators and generate signals
            df_with_indicators = self.signal_generator.calculate_all_indicators(df, htf_df)
            signal_info = self.signal_generator.generate_signals(df_with_indicators)
            
            # Log signal
            if signal_info['signal'] != 'NONE':
                logger.info(
                    f"Signal: {signal_info['signal']} | "
                    f"Price: {signal_info['price']:.2f} | "
                    f"Reason: {signal_info['reason']}"
                )
            
            # Execute trading logic
            await self._execute_trading_logic(signal_info, df_with_indicators)
            
            # Update state
            await self._update_state()
            
        except Exception as e:
            logger.error(f"Trading cycle error: {str(e)}", exc_info=True)
    
    async def _execute_trading_logic(self, signal_info: Dict, df: pd.DataFrame):
        """Execute trading decisions based on signals."""
        signal = signal_info['signal']
        current_price = signal_info['price']
        
        # Get current position
        position = await self.trading_engine.get_position()
        current_contracts = position['contracts']
        current_side = position['side']
        
        # Handle EXIT signals
        if signal == 'EXIT_LONG' and current_side == 'long':
            logger.info("Executing early exit for LONG position")
            await self.trading_engine.close_position()
            self.state.position_side = PositionSide.NONE
            return
        
        if signal == 'EXIT_SHORT' and current_side == 'short':
            logger.info("Executing early exit for SHORT position")
            await self.trading_engine.close_position()
            self.state.position_side = PositionSide.NONE
            return
        
        # Handle BUY signals
        if signal in ['BUY', 'STRONG_BUY']:
            if current_side == 'none' or current_side == 'short':
                # Close short if open
                if current_side == 'short':
                    await self.trading_engine.close_position()
                
                # Open long position
                size = self._calculate_position_size(current_price)
                order = await self.trading_engine.place_market_order('buy', size)
                
                if order:
                    self.state.position_side = PositionSide.LONG
                    self.state.entry_price = current_price
                    self.state.trailing_stop = signal_info['trailing_stop']
                    self.state.target_price = signal_info['target_price_long']
                    
                    # Place take profit order
                    if self.state.target_price:
                        tp_order = await self.trading_engine.place_limit_order(
                            'sell', size, self.state.target_price
                        )
                        if tp_order:
                            self.take_profit_order_id = tp_order['id']
                    
                    logger.info(
                        f"LONG position opened: {size} contracts @ {current_price:.2f}, "
                        f"TP: {self.state.target_price:.2f}, SL: {self.state.trailing_stop:.2f}"
                    )
        
        # Handle SELL signals
        elif signal in ['SELL', 'STRONG_SELL']:
            if current_side == 'none' or current_side == 'long':
                # Close long if open
                if current_side == 'long':
                    await self.trading_engine.close_position()
                
                # Open short position
                size = self._calculate_position_size(current_price)
                order = await self.trading_engine.place_market_order('sell', size)
                
                if order:
                    self.state.position_side = PositionSide.SHORT
                    self.state.entry_price = current_price
                    self.state.trailing_stop = signal_info['trailing_stop']
                    self.state.target_price = signal_info['target_price_short']
                    
                    # Place take profit order
                    if self.state.target_price:
                        tp_order = await self.trading_engine.place_limit_order(
                            'buy', size, self.state.target_price
                        )
                        if tp_order:
                            self.take_profit_order_id = tp_order['id']
                    
                    logger.info(
                        f"SHORT position opened: {size} contracts @ {current_price:.2f}, "
                        f"TP: {self.state.target_price:.2f}, SL: {self.state.trailing_stop:.2f}"
                    )
        
        # Update trailing stop for existing positions
        if current_side != 'none':
            await self._update_trailing_stop(current_price, signal_info['trailing_stop'])
    
    async def _update_trailing_stop(self, current_price: float, new_stop: float):
        """Update trailing stop for existing position."""
        position = await self.trading_engine.get_position()
        
        if position['contracts'] == 0:
            return
        
        # Check if stop loss should be triggered
        if position['side'] == 'long' and current_price <= new_stop:
            logger.info(f"Trailing stop hit for LONG @ {new_stop:.2f}")
            await self.trading_engine.close_position()
            self.state.position_side = PositionSide.NONE
        
        elif position['side'] == 'short' and current_price >= new_stop:
            logger.info(f"Trailing stop hit for SHORT @ {new_stop:.2f}")
            await self.trading_engine.close_position()
            self.state.position_side = PositionSide.NONE
    
    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on capital and risk."""
        # Simplified calculation - in production, add proper risk management
        notional_value = self.config.initial_capital * (self.config.position_size_pct / 100)
        contracts = notional_value / price
        
        # Round to exchange precision
        return round(contracts, 4)
    
    async def _update_state(self):
        """Update internal state from exchange."""
        position = await self.trading_engine.get_position()
        
        if position['contracts'] > 0:
            if position['side'] == 'long':
                self.state.position_side = PositionSide.LONG
            else:
                self.state.position_side = PositionSide.SHORT
        else:
            self.state.position_side = PositionSide.NONE
    
    def _log_final_stats(self):
        """Log final trading statistics."""
        logger.info("=" * 60)
        logger.info("FINAL TRADING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total Trades: {self.state.total_trades}")
        logger.info(f"Winning Trades: {self.state.winning_trades}")
        logger.info(f"Losing Trades: {self.state.losing_trades}")
        if self.state.total_trades > 0:
            win_rate = (self.state.winning_trades / self.state.total_trades) * 100
            logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Total PnL: {self.state.total_pnl:.2f} USDT")
        logger.info("=" * 60)


# ============ RUNNER ============
async def main():
    """Main entry point."""
    # Configure strategy parameters (matching Pine Script defaults)
    config = TradeConfig(
        # UT Bot
        key_value=2.0,
        atr_period=10,
        htf_filter_on=False,
        htf_timeframe="60m",
        vol_filter_on=False,
        vol_len=20,
        
        # DEMA
        dema_len=21,
        
        # Early Exit
        rr_multiplier=1.0,
        use_dema_exit=True,
        use_rsi_exit=True,
        rsi_len=14,
        rsi_overbought=70,
        rsi_oversold=30,
        
        # FVG & Momentum
        use_fvg_filter=True,
        use_adx_filter=True,
        adx_len=14,
        adx_thresh=20,
        body_ratio=0.6,
        
        # Risk
        initial_capital=2000.0,
        position_size_pct=100.0,
        
        # OKX
        symbol="BTC/USDT",
        timeframe="5m",
        leverage=1
    )
    
    # Create and run bot
    bot = Rabiah6XBot(config)
    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
