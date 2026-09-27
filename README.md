# Rabiah6X - UT Bot + 1:1 Scalp Targets + DEMA + FVG

Experimental Python implementation of a TradingView Pine Script strategy using OKX exchange APIs.

> **Project status:** educational / research prototype, not a production-ready trading system. The code includes real-order execution paths if configured with exchange credentials. Review the implementation, permissions and risk controls independently before any use. No performance or profitability claim is made here.

> **Credential safety:** only `.env.example` is tracked. Copy it to a local, git-ignored `.env`; never commit real API keys. Exchange keys with withdrawal permissions should never be used for experiments.

## Features

- **UT Bot Trailing Stop**: Core ATR-based trailing stop mechanism
- **DEMA (Double Exponential Moving Average)**: Fast exit signals
- **RSI Exit**: Overbought/oversold reversal exits
- **FVG (Fair Value Gap) Detection**: Identifies market inefficiencies for strong signals
- **ADX Trend Filter**: Confirms strong trend conditions
- **Expansion Bar Detection**: Identifies momentum candles
- **1:1 Risk:Reward Targets**: Scalper-friendly take-profit levels
- **Higher Timeframe Filter**: Optional trend confirmation from HTF
- **Volume Filter**: Optional volume confirmation

## Installation

```bash
pip install okx ccxt python-dotenv pandas numpy
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your OKX API credentials:
```
OKX_API_KEY=your_api_key_here
OKX_API_SECRET=your_api_secret_here
OKX_PASSPHRASE=your_passphrase_here
```

## Usage

### Run in Simulation Mode (no real trades)
Without API credentials, the bot runs in simulation mode with mock data:

```bash
python okx_rabiah6x_strategy.py
```

### Run with Live Trading
With valid API credentials, the bot connects to OKX and executes real trades:

```bash
python okx_rabiah6x_strategy.py
```

## Strategy Parameters

All parameters match the original Pine Script strategy:

### UT Bot Settings
- `key_value`: ATR multiplier for trailing stop (default: 2.0)
- `atr_period`: ATR calculation period (default: 10)
- `htf_filter_on`: Enable higher timeframe trend filter (default: False)
- `htf_timeframe`: Higher timeframe for trend filter (default: "60m")
- `vol_filter_on`: Enable volume filter (default: False)
- `vol_len`: Volume MA length (default: 20)

### DEMA Settings
- `dema_len`: DEMA length (default: 21)

### Early Exit Settings
- `rr_multiplier`: Risk:Reward target (default: 1.0 = 1:1)
- `use_dema_exit`: Enable DEMA fast exit (default: True)
- `use_rsi_exit`: Enable RSI reversal exit (default: True)
- `rsi_len`: RSI length (default: 14)
- `rsi_overbought`: RSI overbought level for long exit (default: 70)
- `rsi_oversold`: RSI oversold level for short exit (default: 30)

### FVG & Momentum Settings
- `use_fvg_filter`: Track FVG confluence (default: True)
- `use_adx_filter`: Require ADX strong trend (default: True)
- `adx_len`: ADX length (default: 14)
- `adx_thresh`: ADX strong trend threshold (default: 20)
- `body_ratio`: Expansion candle body/range ratio (default: 0.6)

### Risk Management
- `initial_capital`: Starting capital (default: 2000 USDT)
- `position_size_pct`: Position size as % of equity (default: 100%)
- `symbol`: Trading pair (default: "BTC/USDT")
- `timeframe`: Chart timeframe (default: "5m")
- `leverage`: Futures leverage (default: 1)

## Signal Types

The bot generates the following signals:

- **STRONG_BUY**: UT Bot crossover + FVG + Strong Trend + Expansion Bar
- **BUY**: Standard UT Bot crossover
- **STRONG_SELL**: UT Bot crossunder + FVG + Strong Trend + Expansion Bar
- **SELL**: Standard UT Bot crossunder
- **EXIT_LONG**: Early exit for long positions (DEMA/RSI trigger)
- **EXIT_SHORT**: Early exit for short positions (DEMA/RSI trigger)

## Logging

The bot includes a comprehensive logging system:

- **Console logs**: INFO level and above
- **File logs**: DEBUG level in `trading_bot.log`

Log format includes timestamp, level, module, function, line number, and message.

## Architecture

```
okx_rabiah6x_strategy.py
├── setup_logging()           # Logging configuration
├── TradeConfig               # Strategy configuration dataclass
├── TradeState                # Current trading state
├── TechnicalIndicators       # Indicator calculations
│   ├── ema()
│   ├── dema()
│   ├── atr()
│   ├── rsi()
│   ├── adx()
│   ├── detect_fvg()
│   └── detect_expansion_bar()
├── UTBotCalculator           # UT Bot trailing stop logic
├── SignalGenerator           # Signal generation logic
├── OKXTradingEngine          # Exchange operations
│   ├── connect()
│   ├── get_ohlcv()
│   ├── place_market_order()
│   ├── place_limit_order()
│   └── close_position()
└── Rabiah6XBot               # Main bot orchestrator
```

## Safety Features

1. **Simulation Mode**: Runs without API credentials for testing
2. **Error Handling**: Comprehensive try-catch blocks throughout
3. **Rate Limiting**: Built-in rate limiting via CCXT
4. **Position Tracking**: Maintains accurate position state
5. **Trailing Stop Updates**: Continuously monitors and updates stops

## Important Notes

⚠️ **Risk Warning**: This bot is for educational purposes. Cryptocurrency trading involves significant risk. Always test thoroughly before using real funds.

⚠️ **API Security**: Never commit your `.env` file with real credentials to version control.

⚠️ **Backtesting**: The original Pine Script strategy should be backtested on TradingView before live deployment.

## Requirements

- Python 3.8+
- okx >= 2.1.0
- ccxt >= 4.0.0
- pandas >= 1.5.0
- numpy >= 1.20.0
- python-dotenv >= 0.19.0

## License

This code is provided as-is for educational purposes. Use at your own risk.
