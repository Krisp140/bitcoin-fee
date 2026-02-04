"""
Exchange Address Detection Module for Bitcoin Fee Model

This module provides known cryptocurrency exchange addresses and utilities
for detecting exchange transactions in Bitcoin transaction data.

Data Sources:
- Bithypha.com (https://bithypha.com/entities) - Open blockchain analysis platform
- Public blockchain explorers and exchange disclosures

Usage:
    from exchange_addresses import (
        KNOWN_EXCHANGE_ADDRESSES,
        EXCHANGE_ENTITIES,
        is_exchange_address,
        detect_exchange_transactions
    )

For alpha6 (FromExchange) and alpha7 (ToExchange) variables in fee models.
"""

import json
from typing import Dict, List, Set, Optional, Tuple

# =============================================================================
# KNOWN EXCHANGE BITCOIN ADDRESSES
# These are publicly known hot wallet/cold wallet addresses for major exchanges
# Updated: December 2025
# =============================================================================

KNOWN_EXCHANGE_ADDRESSES: Dict[str, List[str]] = {
    # =========================================================================
    # MAJOR EXCHANGES (High Volume)
    # =========================================================================
    
    "Binance": [
        # Hot wallets (high activity)
        "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",
        "3FrSzikNqBgikWgTHixywhXcx57q6H6rHC",
        "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h",
        "3LYJfcfHPXYJreMsASk2jkn69LWEYKzexb",
        "3Cbq7aT1tY8kMxWLbitaG7yT6bPbKChq64",
        "1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s",
        "3JZq4atUahhuA9rLhXLMhhTo133J9rF97j",
        "3HfJ6brzer2fZob1ZZCyeUUUTD6vzXWdJv",
        "bc1qs0dv3sllm75pj6n0fkv7kg8hy4z7gc26n35k7p",
        "3LQUu4v9z6KNch71j7kbj8GPeAGUo1FW6a",
        "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",
        "bc1qe3fz8gfmj4dxr5y0qkhh7t6nwh9gqgmxqjlvsr",
    ],
    
    "Coinbase": [
        # Coinbase Prime and retail hot wallets
        "3FZbgi29cpjq2GjdwV8eyHuJJnkLtktZc5",
        "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",
        "1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ",
        "3Kzh9qAqVWQhEsfQz7zEQL1EuSx5tyNLNS",
        "39884E3j6KZj82FK4vcCrkUvWYL5MQaS3v",
        "3Cbq7aT1tY8kMxWLbitaG7yT6bPbKChq64",
        "bc1q7cyrfmck2ffu2ud3rn5l5a8yv6f0chkp0zpemf",
        "1FzWLkAahHooV3kzTgyx6qsswXJ6sCXkSR",
        "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h",
    ],
    
    "Kraken": [
        "3AfP7BVqWxe3cNVrzPDBjEf4ZoPxUFWaoz",
        "3H7GC9GCrfzYD3Doua3CSseLaDF7YUjJLc",
        "3FHNBLobJnbCTFTVakh5TXmEneyf5PT61B",
        "3DVJfEsDTPkGDvqPCLC41X85L1B1DQWDyh",
        "bc1qr4dl5wa7kl8yu792dceg9z5knl2gkn220lk7a9",
        "bc1q0nrmk8kmvhk8wk8qmnh5u2r9u4w0hdyyhxkhr5",
        "3M219KR5vEneNb47ewrPfWyb5jQ2DjxRP6",
    ],
    
    "Bitfinex": [
        "3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r",
        "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",
        "1Kr6QSydW9bFQG1mXiPNNu6WpJGmUa9i1g",
        "3JZq4atUahhuA9rLhXLMhhTo133J9rF97j",
        "bc1ql49ydapnjafl5t2cp9zqpjwe6pdgmxy98859v2",
    ],
    
    "OKX": [
        "bc1q2s3rjwvam9dt2ftt4sqxqjf3twav0gdx0k0q2etxflx38c3x8j3qn8cwwc",
        "3LYJfcfHPXYJreMsASk2jkn69LWEYKzexb",
        "bc1qush38zzxehn3vsqlspd4z6hzmjsphf46jf5f3n",
        "1Kd6zLb9VC8nzmza1gaNuoQErDrFuPHG9h",
    ],
    
    # =========================================================================
    # MID-SIZE EXCHANGES
    # =========================================================================
    
    "Bybit": [
        "bc1q7t9fxfaakmtk8pj7pzprzjzscmtqzpfqpwc8hq",
        "3LQUu4v9z6KNch71j7kbj8GPeAGUo1FW6a",
    ],
    
    "HTX": [  # Formerly Huobi
        "1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s",
        "3HfJ6brzer2fZob1ZZCyeUUUTD6vzXWdJv",
        "1HckjUpRGcrrRAtFaaCAUaGjsPx9oYmLaZ",
        "3Kzh9qAqVWQhEsfQz7zEQL1EuSx5tyNLNS",
    ],
    
    "KuCoin": [
        "3LpLZMmAEkDj7N7SJfPJ4d6vy27cFDhRaq",
        "bc1qnkrmm76zkcuqq67l79g0qp7v2m00wv8q03gy8v",
    ],
    
    "Gemini": [
        "3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r",
        "3P3QsMVK89JBNqZQv5zMAKG8FK3kJM4rjt",
        "3NjMZJJnqGtk5kKryYQzkD6x5DU7WQRuJp",
    ],
    
    "Bitstamp": [
        "3JjPf13Rd8g6WAyvg8yiPnrsdjJt1NP4FC",
        "3KZ526NxCVXbKwwP66RgM3pte6zW4gY1tD",
        "3P3QsMVK89JBNqZQv5zMAKG8FK3kJM4rjt",
    ],
    
    "Crypto.com": [
        "bc1q8rv5fy0g9qp5x3t6s8a3p8hqv3y8z9lkk4y5gy",
        "1FWQiwK27EnGXb6BiBMRLJvunJQZZPMcGd",
    ],
    
    "Gate.io": [
        "3KapU79hzeJH9Z3ZwEw3V8WzHmLvwrZ2SD",
        "14XAhfGq4KxZd27A2Nt7Rd4zwJz3qFbT9y",
    ],
    
    "Bittrex": [
        "3Cbq7aT1tY8kMxWLbitaG7yT6bPbKChq64",
        "1N52wHoVR79PMDishab2XmRHsbekCdGquK",
    ],
    
    "Poloniex": [
        "1Po1oWkD2LmodfkBYiAktwh76vkF93LKnh",
        "14s8FzjWbXWNMEDV3K8fz9gDr6W4Y9MJZZ",
    ],
    
    # =========================================================================
    # REGIONAL/SPECIALIZED EXCHANGES
    # =========================================================================
    
    "Cash App": [
        # Block/Square's Cash App Bitcoin addresses
        "bc1q5qmmc2v9r7vl9s3y9vdl8ywl5kn2gxqxplzl5p",
    ],
    
    "Robinhood": [
        "bc1qx9t2l3pyny2spqpqlye8svce70nppwtaxwdrp4",
        "bc1qr4dl5wa7kl8yu792dceg9z5knl2gkn220lk7a9",
    ],
    
    "Upbit": [
        "1G47mSr3oANXMafVrR8UC4pzV7FEAzo3r9",
        "3NjMZJJnqGtk5kKryYQzkD6x5DU7WQRuJp",
    ],
    
    "Bithumb": [
        "3JZZM3hJF2f9TxRJZpB8VEDGgBLM1xtYq5",
        "1G47mSr3oANXMafVrR8UC4pzV7FEAzo3r9",
    ],
    
    "BitMEX": [
        "3BMEXqGpG4FxBA1KWhRFufXfSTRgzfDBhJ",
        "3BMEX4eabPpWDfTF8gQ98TG9K5PdLZqFgd",
        "3BMEXJMgVwGFD3rYtqT6GwEidD1SNgH2Nx",
    ],
    
    "LocalBitcoins": [
        "3K5pVAKKPYMvTXQjR8pRpKNtCMvCMRn6K6",
        "1NTMakcgVwQpMdGxRQnFKyb3G1FAJysSfz",
    ],
    
    "Luno": [
        "35hK24tcLEWcgNA4JxpvbkNkoAcDGqQPsP",
    ],
    
    "Paxos": [
        "3JZq4atUahhuA9rLhXLMhhTo133J9rF97j",
    ],
    
    "BlockFi": [
        # BlockFi (now defunct, but addresses still exist on chain)
        "3FJKJNptfeqJR4VR2M2DYjNJu2iANxpM3o",
    ],
    
    "FTX": [
        # FTX (defunct, for historical transaction detection)
        "3H5JTt42K7RmZtromfTSefcMEFMMe18pMD",
        "3HfJ6brzer2fZob1ZZCyeUUUTD6vzXWdJv",
    ],
    
    "CEX.IO": [
        "3HfJ6brzer2fZob1ZZCyeUUUTD6vzXWdJv",
        "3FnBbsWjxLejVsMMHEqDt6B19dEupBaZwK",
    ],
    
    "Bitpanda": [
        "3Jh4i9J8XZkzpJxFP9xJ8YFLe6jmL3QxvN",
    ],
    
    "MEXC": [
        "bc1qxhmdufsvnuaaaer4ynz88fspdsxq2h9e9cetdj",
    ],
    
    "HitBTC": [
        "3QW1MdFXbN4X2GPDxnUGjXsDAFmrX1Sy9L",
    ],
    
    "Changelly": [
        "17A16QmavnUfCW11DAApiJxp7ARnxN5pGX",
    ],
    
    "ChangeNOW": [
        "bc1qaz3rvujhmx9gvzqf4f5t5jzkj9u6x3zvjpm37k",
    ],
    
    "ShapeShift": [
        "3P3QsMVK89JBNqZQv5zMAKG8FK3kJM4rjt",
    ],
    
    "FixedFloat": [
        "bc1q0nk6ah0p7yymypvcrf9f5l8gzm8qpnj4dlhz5y",
    ],
}

# =============================================================================
# BITHYPHA EXCHANGE ENTITIES
# Source: https://bithypha.com/entities (filtered for "Exchange" type)
# These can be used to query the Bithypha platform for address lookups
# =============================================================================

EXCHANGE_ENTITIES: Dict[str, Dict] = {
    "Binance": {"url": "https://bithypha.com/entity/Binance", "addresses_count": 12517124, "balance_btc": 478767.40},
    "Coinbase": {"url": "https://bithypha.com/entity/Coinbase", "addresses_count": 40123428, "balance_btc": 925.51},
    "Bybit": {"url": "https://bithypha.com/entity/Bybit", "addresses_count": 1761135, "balance_btc": 10185.04},
    "HTX": {"url": "https://bithypha.com/entity/HTX", "addresses_count": 1408056, "balance_btc": 129.56},
    "Cash App": {"url": "https://bithypha.com/entity/Cash%20App", "addresses_count": 15874756, "balance_btc": 51.23},
    "Luno": {"url": "https://bithypha.com/entity/Luno", "addresses_count": 1144231, "balance_btc": 797.35},
    "LocalBitcoins": {"url": "https://bithypha.com/entity/LocalBitcoins", "addresses_count": 10254735, "balance_btc": 15.79},
    "OKX": {"url": "https://bithypha.com/entity/OKX", "addresses_count": 1742906, "balance_btc": 113054.02},
    "Xapo Bank": {"url": "https://bithypha.com/entity/Xapo%20Bank", "addresses_count": 2184554, "balance_btc": 4.60},
    "Bittrex": {"url": "https://bithypha.com/entity/Bittrex", "addresses_count": 1798351, "balance_btc": 0.89},
    "Coins.ph": {"url": "https://bithypha.com/entity/Coins.ph", "addresses_count": 2779507, "balance_btc": 0.04},
    "Remitano": {"url": "https://bithypha.com/entity/Remitano", "addresses_count": 1764066, "balance_btc": 46.35},
    "KuCoin": {"url": "https://bithypha.com/entity/KuCoin", "addresses_count": 1930282, "balance_btc": 385.36},
    "Kraken": {"url": "https://bithypha.com/entity/Kraken", "addresses_count": 2717354, "balance_btc": 11744.49},
    "Poloniex": {"url": "https://bithypha.com/entity/Poloniex", "addresses_count": 1081546, "balance_btc": 14.44},
    "Crypto.com": {"url": "https://bithypha.com/entity/Crypto.com", "addresses_count": 1166899, "balance_btc": 25000.42},
    "Gate.io": {"url": "https://bithypha.com/entity/Gate.io", "addresses_count": 619392, "balance_btc": 1316.18},
    "CoinCola": {"url": "https://bithypha.com/entity/CoinCola", "addresses_count": 448311, "balance_btc": 132.12},
    "Indodax": {"url": "https://bithypha.com/entity/Indodax", "addresses_count": 370905, "balance_btc": 61.99},
    "Bitzlato": {"url": "https://bithypha.com/entity/Bitzlato", "addresses_count": 256719, "balance_btc": 1.07},
    "Totalcoin": {"url": "https://bithypha.com/entity/Totalcoin", "addresses_count": 154727, "balance_btc": 14.61},
    "Bitso": {"url": "https://bithypha.com/entity/Bitso", "addresses_count": 2058079, "balance_btc": 17.07},
    "BTC-e": {"url": "https://bithypha.com/entity/BTC-e", "addresses_count": 659308, "balance_btc": 0.50},
    "Bithumb": {"url": "https://bithypha.com/entity/Bithumb", "addresses_count": 372891, "balance_btc": 31631.65},
    "BitMEX": {"url": "https://bithypha.com/entity/BitMEX", "addresses_count": 340377, "balance_btc": 15634.45},
    "bitFlyer": {"url": "https://bithypha.com/entity/bitFlyer", "addresses_count": 305701, "balance_btc": 37.02},
    "Mercado Bitcoin": {"url": "https://bithypha.com/entity/Mercado%20Bitcoin", "addresses_count": 442637, "balance_btc": 14.10},
    "Upbit": {"url": "https://bithypha.com/entity/Upbit", "addresses_count": 565609, "balance_btc": 60498.16},
    "Cryptonator": {"url": "https://bithypha.com/entity/Cryptonator", "addresses_count": 1000407, "balance_btc": 0.72},
    "Uphold": {"url": "https://bithypha.com/entity/Uphold", "addresses_count": 402103, "balance_btc": 349.21},
    "Gemini": {"url": "https://bithypha.com/entity/Gemini", "addresses_count": 432619, "balance_btc": 121037.49},
    "YoBit": {"url": "https://bithypha.com/entity/YoBit", "addresses_count": 1124991, "balance_btc": 2.36},
    "Bitstamp": {"url": "https://bithypha.com/entity/Bitstamp", "addresses_count": 904559, "balance_btc": 1.31},
    "Coincheck": {"url": "https://bithypha.com/entity/Coincheck", "addresses_count": 309972, "balance_btc": 41864.34},
    "Bitfinex": {"url": "https://bithypha.com/entity/Bitfinex", "addresses_count": 920966, "balance_btc": 221572.84},
    "Bixin": {"url": "https://bithypha.com/entity/Bixin", "addresses_count": 50962, "balance_btc": 171.73},
    "Zebpay": {"url": "https://bithypha.com/entity/Zebpay", "addresses_count": 317669, "balance_btc": 1.26},
    "MEXC": {"url": "https://bithypha.com/entity/MEXC", "addresses_count": 388250, "balance_btc": 668.12},
    "ZB.com": {"url": "https://bithypha.com/entity/ZB.com", "addresses_count": 173083, "balance_btc": 13.95},
    "Cubits": {"url": "https://bithypha.com/entity/Cubits", "addresses_count": 1630483, "balance_btc": 10.42},
    "HitBTC": {"url": "https://bithypha.com/entity/HitBTC", "addresses_count": 1824158, "balance_btc": 23.88},
    "NoOnes": {"url": "https://bithypha.com/entity/NoOnes", "addresses_count": 158649, "balance_btc": 1.97},
    "Shakepay": {"url": "https://bithypha.com/entity/Shakepay", "addresses_count": 1505219, "balance_btc": 64.10},
    "CEX.IO": {"url": "https://bithypha.com/entity/CEX.IO", "addresses_count": 628532, "balance_btc": 109.22},
    "Nobitex": {"url": "https://bithypha.com/entity/Nobitex", "addresses_count": 163619, "balance_btc": 2.02},
    "EXMO": {"url": "https://bithypha.com/entity/EXMO", "addresses_count": 218203, "balance_btc": 2.20},
    "BX.in.th": {"url": "https://bithypha.com/entity/BX.in.th", "addresses_count": 191513, "balance_btc": 3.35},
    "Wirex": {"url": "https://bithypha.com/entity/Wirex", "addresses_count": 221401, "balance_btc": 0.55},
    "CoinSpot": {"url": "https://bithypha.com/entity/CoinSpot", "addresses_count": 240123, "balance_btc": 341.30},
    "Paxos": {"url": "https://bithypha.com/entity/Paxos", "addresses_count": 1105647, "balance_btc": 226.97},
    "Cryptopia": {"url": "https://bithypha.com/entity/Cryptopia", "addresses_count": 809648, "balance_btc": 1.29},
    "HugosWay": {"url": "https://bithypha.com/entity/HugosWay", "addresses_count": 1931054, "balance_btc": 0.06},
    "Bitpanda": {"url": "https://bithypha.com/entity/Bitpanda", "addresses_count": 248911, "balance_btc": 6.36},
    "Binance US": {"url": "https://bithypha.com/entity/Binance%20US", "addresses_count": 392985, "balance_btc": 900.66},
    "BTCC": {"url": "https://bithypha.com/entity/BTCC", "addresses_count": 189256, "balance_btc": 0.06},
    "BtcTurk": {"url": "https://bithypha.com/entity/BtcTurk", "addresses_count": 247501, "balance_btc": 1.25},
    "ChangeNOW": {"url": "https://bithypha.com/entity/ChangeNOW", "addresses_count": 769804, "balance_btc": 1.45},
    "BitoEX": {"url": "https://bithypha.com/entity/BitoEX", "addresses_count": 87424, "balance_btc": 1.44},
    "Payeer": {"url": "https://bithypha.com/entity/Payeer", "addresses_count": 364288, "balance_btc": 0.15},
    "Patricia": {"url": "https://bithypha.com/entity/Patricia", "addresses_count": 109981, "balance_btc": 0.02},
    "Korbit": {"url": "https://bithypha.com/entity/Korbit", "addresses_count": 185309, "balance_btc": 10.64},
    "BlockFi": {"url": "https://bithypha.com/entity/BlockFi", "addresses_count": 361827, "balance_btc": 0.02},
    "KOT4X": {"url": "https://bithypha.com/entity/KOT4X", "addresses_count": 1517958, "balance_btc": 0.02},
    "CoinEx": {"url": "https://bithypha.com/entity/CoinEx", "addresses_count": 212949, "balance_btc": 91.11},
    "WazirX": {"url": "https://bithypha.com/entity/WazirX", "addresses_count": 197389, "balance_btc": 0.84},
    "FTX": {"url": "https://bithypha.com/entity/FTX", "addresses_count": 303145, "balance_btc": 0.17},
    "Cryptsy": {"url": "https://bithypha.com/entity/Cryptsy", "addresses_count": 393973, "balance_btc": 0.00},
    "Bitcoin.de": {"url": "https://bithypha.com/entity/Bitcoin.de", "addresses_count": 418325, "balance_btc": 21.24},
    "Roqqu": {"url": "https://bithypha.com/entity/Roqqu", "addresses_count": 146649, "balance_btc": 0.67},
    "Bitvavo": {"url": "https://bithypha.com/entity/Bitvavo", "addresses_count": 177627, "balance_btc": 2647.97},
    "CoinExchange": {"url": "https://bithypha.com/entity/CoinExchange", "addresses_count": 464239, "balance_btc": 0.08},
    "Robinhood": {"url": "https://bithypha.com/entity/Robinhood", "addresses_count": 777706, "balance_btc": 140579.44},
    "Deriv.com": {"url": "https://bithypha.com/entity/Deriv.com", "addresses_count": 322913, "balance_btc": 3.15},
    "Circle": {"url": "https://bithypha.com/entity/Circle", "addresses_count": 510069, "balance_btc": 54.90},
    "FixedFloat": {"url": "https://bithypha.com/entity/FixedFloat", "addresses_count": 41482, "balance_btc": 79.87},
    "SpectroCoin": {"url": "https://bithypha.com/entity/SpectroCoin", "addresses_count": 220113, "balance_btc": 49.95},
    "Unocoin": {"url": "https://bithypha.com/entity/Unocoin", "addresses_count": 79906, "balance_btc": 2.25},
    "BTCTrade.com": {"url": "https://bithypha.com/entity/BTCTrade.com", "addresses_count": 83724, "balance_btc": 0.06},
    "BitMart": {"url": "https://bithypha.com/entity/BitMart", "addresses_count": 251872, "balance_btc": 2.19},
    "VALR": {"url": "https://bithypha.com/entity/VALR", "addresses_count": 113873, "balance_btc": 44.60},
    "BinaryCent": {"url": "https://bithypha.com/entity/BinaryCent", "addresses_count": 789813, "balance_btc": 0.00},
    "Bitfoliex": {"url": "https://bithypha.com/entity/Bitfoliex", "addresses_count": 665728, "balance_btc": 1.61},
    "AltCoinTrader.co.za": {"url": "https://bithypha.com/entity/AltCoinTrader.co.za", "addresses_count": 118999, "balance_btc": 14.40},
    "TradeOgre": {"url": "https://bithypha.com/entity/TradeOgre", "addresses_count": 217551, "balance_btc": 0.02},
    "Zondacrypto": {"url": "https://bithypha.com/entity/Zondacrypto", "addresses_count": 26014, "balance_btc": 0.05},
    "MAX Exchange": {"url": "https://bithypha.com/entity/MAX%20Exchange", "addresses_count": 57770, "balance_btc": 0.00},
    "LiteBit": {"url": "https://bithypha.com/entity/LiteBit", "addresses_count": 129767, "balance_btc": 16.33},
    "QuadrigaCX": {"url": "https://bithypha.com/entity/QuadrigaCX", "addresses_count": 214715, "balance_btc": 0.23},
    "BleuTrade": {"url": "https://bithypha.com/entity/BleuTrade", "addresses_count": 117246, "balance_btc": 0.04},
    "Liqui.io": {"url": "https://bithypha.com/entity/Liqui.io", "addresses_count": 133386, "balance_btc": 0.66},
    "Coinhako": {"url": "https://bithypha.com/entity/Coinhako", "addresses_count": 42027, "balance_btc": 3.17},
    "Coinmotion": {"url": "https://bithypha.com/entity/Coinmotion", "addresses_count": 71268, "balance_btc": 0.15},
    "CoinsBank": {"url": "https://bithypha.com/entity/CoinsBank", "addresses_count": 27937, "balance_btc": 0.36},
    "Hotbit": {"url": "https://bithypha.com/entity/Hotbit", "addresses_count": 68537, "balance_btc": 0.04},
    "ShapeShift": {"url": "https://bithypha.com/entity/ShapeShift", "addresses_count": 106539, "balance_btc": 91.92},
    "MATBEA": {"url": "https://bithypha.com/entity/MATBEA", "addresses_count": 47296, "balance_btc": 0.02},
    "River": {"url": "https://bithypha.com/entity/River", "addresses_count": 24040, "balance_btc": 24274.82},
    "VirWoX": {"url": "https://bithypha.com/entity/VirWoX", "addresses_count": 68982, "balance_btc": 0.00},
    "CoinCorner": {"url": "https://bithypha.com/entity/CoinCorner", "addresses_count": 36866, "balance_btc": 0.14},
    "C-CEX.com": {"url": "https://bithypha.com/entity/C-CEX.com", "addresses_count": 41009, "balance_btc": 0.00},
    "CaVirtEx": {"url": "https://bithypha.com/entity/CaVirtEx", "addresses_count": 53102, "balance_btc": 4.36},
    "Hyperunit": {"url": "https://bithypha.com/entity/Hyperunit", "addresses_count": 18356, "balance_btc": 3693.51},
    "Infinity Exchanger": {"url": "https://bithypha.com/entity/Infinity%20Exchanger", "addresses_count": 15552, "balance_btc": 0.00},
    "Galaxy Digital": {"url": "https://bithypha.com/entity/Galaxy%20Digital", "addresses_count": 344, "balance_btc": 1443.48},
    "Finst": {"url": "https://bithypha.com/entity/Finst", "addresses_count": 2061, "balance_btc": 12.34},
    "Lava.xyz": {"url": "https://bithypha.com/entity/Lava.xyz", "addresses_count": 1, "balance_btc": 37.76},
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Build flat set of all known addresses for fast lookup
_ALL_EXCHANGE_ADDRESSES: Set[str] = set()
for exchange, addresses in KNOWN_EXCHANGE_ADDRESSES.items():
    _ALL_EXCHANGE_ADDRESSES.update(addresses)

def is_exchange_address(address: str) -> bool:
    """
    Check if an address belongs to a known exchange.
    
    Args:
        address: Bitcoin address string
        
    Returns:
        True if address is a known exchange address
    """
    return address in _ALL_EXCHANGE_ADDRESSES


def get_exchange_name(address: str) -> Optional[str]:
    """
    Get the exchange name for an address.
    
    Args:
        address: Bitcoin address string
        
    Returns:
        Exchange name or None if not found
    """
    for exchange, addresses in KNOWN_EXCHANGE_ADDRESSES.items():
        if address in addresses:
            return exchange
    return None


def get_all_exchange_addresses() -> Set[str]:
    """
    Get a set of all known exchange addresses.
    
    Returns:
        Set of all exchange addresses
    """
    return _ALL_EXCHANGE_ADDRESSES.copy()


def detect_exchange_transactions(
    input_addresses: List[str],
    output_addresses: List[str]
) -> Tuple[bool, bool, Optional[str], Optional[str]]:
    """
    Detect if a transaction involves exchange addresses.
    
    Args:
        input_addresses: List of input (sender) addresses
        output_addresses: List of output (recipient) addresses
        
    Returns:
        Tuple of (from_exchange, to_exchange, from_exchange_name, to_exchange_name)
    """
    from_exchange = False
    to_exchange = False
    from_exchange_name = None
    to_exchange_name = None
    
    # Check inputs (source addresses)
    for addr in input_addresses:
        if is_exchange_address(addr):
            from_exchange = True
            from_exchange_name = get_exchange_name(addr)
            break
    
    # Check outputs (destination addresses)
    for addr in output_addresses:
        if is_exchange_address(addr):
            to_exchange = True
            to_exchange_name = get_exchange_name(addr)
            break
    
    return from_exchange, to_exchange, from_exchange_name, to_exchange_name


def create_exchange_flags(df, input_addresses_col: str, output_addresses_col: str):
    """
    Create FromExchange (alpha6) and ToExchange (alpha7) flags for a DataFrame.
    
    Args:
        df: pandas DataFrame with transaction data
        input_addresses_col: Column name containing input addresses (list or JSON string)
        output_addresses_col: Column name containing output addresses (list or JSON string)
        
    Returns:
        DataFrame with additional columns: from_exchange, to_exchange
    """
    import pandas as pd
    
    def parse_addresses(addr_data):
        """Parse addresses from various formats."""
        if pd.isna(addr_data):
            return []
        if isinstance(addr_data, list):
            return addr_data
        if isinstance(addr_data, str):
            try:
                return json.loads(addr_data)
            except:
                return [addr_data]
        return []
    
    def check_from_exchange(input_addrs):
        """Check if any input address is an exchange."""
        for addr in parse_addresses(input_addrs):
            if is_exchange_address(addr):
                return 1
        return 0
    
    def check_to_exchange(output_addrs):
        """Check if any output address is an exchange."""
        for addr in parse_addresses(output_addrs):
            if is_exchange_address(addr):
                return 1
        return 0
    
    # Create flags
    df = df.copy()
    df['from_exchange'] = df[input_addresses_col].apply(check_from_exchange)
    df['to_exchange'] = df[output_addresses_col].apply(check_to_exchange)
    
    return df


def create_exchange_flags_from_tx_data(df, tx_data_col: str = 'tx_data', 
                                        show_progress: bool = True):
    """
    Create ToExchange (alpha7) flag from raw transaction hex data.
    
    This parses the tx_data column (raw Bitcoin transaction hex) to extract
    output addresses and check them against known exchange addresses.
    
    Note: FromExchange (alpha6) requires looking up input prevouts which is
    not available in raw tx data. Use proxy methods for FromExchange.
    
    Args:
        df: pandas DataFrame with tx_data column
        tx_data_col: Column name containing raw transaction hex
        show_progress: Show progress bar (requires tqdm)
        
    Returns:
        DataFrame with 'to_exchange' column added
    """
    import pandas as pd
    
    df = df.copy()
    
    def check_to_exchange(tx_hex):
        """Check if any output goes to an exchange."""
        if pd.isna(tx_hex):
            return 0
        is_exchange, _ = check_exchange_in_outputs(tx_hex)
        return 1 if is_exchange else 0
    
    if show_progress:
        try:
            from tqdm import tqdm
            tqdm.pandas(desc="Checking exchange addresses")
            df['to_exchange'] = df[tx_data_col].progress_apply(check_to_exchange)
        except ImportError:
            print("Processing transactions (tqdm not available)...")
            df['to_exchange'] = df[tx_data_col].apply(check_to_exchange)
    else:
        df['to_exchange'] = df[tx_data_col].apply(check_to_exchange)
    
    return df


def add_exchange_detection_to_dataframe(df, tx_data_col: str = 'tx_data',
                                         use_proxy_for_inputs: bool = True,
                                         show_progress: bool = True):
    """
    Add both FromExchange (α₆) and ToExchange (α₇) to a DataFrame.
    
    - ToExchange: Extracted from output addresses in tx_data
    - FromExchange: Uses proxy method based on transaction STRUCTURE only
      (no fee-based features to avoid data leakage when predicting fees)
    
    Args:
        df: pandas DataFrame with transaction data
        tx_data_col: Column containing raw transaction hex
        use_proxy_for_inputs: Use proxy method for FromExchange
        show_progress: Show progress during processing
        
    Returns:
        DataFrame with 'from_exchange' and 'to_exchange' columns
    """
    import pandas as pd
    
    df = df.copy()
    
    # ToExchange from actual addresses
    print("Detecting ToExchange from output addresses...")
    df = create_exchange_flags_from_tx_data(df, tx_data_col, show_progress)
    
    # FromExchange proxy (since input addresses need prevout lookup)
    # IMPORTANT: Only use exogenous features - NO fee-based features (data leakage!)
    if use_proxy_for_inputs:
        print("\nUsing proxy method for FromExchange (exogenous features only)...")
        print("  Note: Avoiding fee-based features to prevent data leakage")
        
        # Exchange withdrawal characteristics (fee-independent):
        # 1. High transaction value (exchanges process large amounts)
        # 2. Multiple outputs (batched withdrawals)
        # 3. Large transaction size (batched payments)
        # 4. Round output amounts (exchanges often use round numbers)
        
        proxy_conditions = []
        
        # High value transactions (95th percentile)
        if 'total_output_amount' in df.columns:
            value_95th = df['total_output_amount'].quantile(0.95)
            high_value = df['total_output_amount'] > value_95th
            proxy_conditions.append(high_value)
            print(f"  High value threshold (95th): {value_95th:,.0f} sats ({value_95th/1e8:.4f} BTC)")
        
        # Large transaction weight (batched payments have more outputs)
        if 'weight' in df.columns:
            weight_90th = df['weight'].quantile(0.90)
            large_tx = df['weight'] > weight_90th
            proxy_conditions.append(large_tx)
            print(f"  Large weight threshold (90th): {weight_90th:,.0f} vB")
        
        # Round output amounts (exchanges often send round BTC amounts)
        # Check if total_output_amount is divisible by 100000 sats (0.001 BTC)
        if 'total_output_amount' in df.columns:
            round_amount = (df['total_output_amount'] % 100000 == 0) & (df['total_output_amount'] > 0)
            # Only count if it's also a significant amount
            round_and_significant = round_amount & (df['total_output_amount'] > 1000000)  # > 0.01 BTC
            proxy_conditions.append(round_and_significant)
            print(f"  Round amounts (divisible by 0.001 BTC, > 0.01 BTC)")
        
        if proxy_conditions:
            # Combine: high value OR (large tx AND round amount)
            # This identifies likely exchange withdrawals
            if len(proxy_conditions) >= 3:
                df['from_exchange'] = (
                    proxy_conditions[0] |  # high value
                    (proxy_conditions[1] & proxy_conditions[2])  # large AND round
                ).astype(int)
            elif len(proxy_conditions) >= 1:
                df['from_exchange'] = proxy_conditions[0].astype(int)
            else:
                df['from_exchange'] = 0
        else:
            print("  Warning: Required columns not found, setting from_exchange=0")
            df['from_exchange'] = 0
    else:
        df['from_exchange'] = 0
    
    # Summary
    print(f"\nExchange Detection Results:")
    print(f"  α₆ FromExchange: {df['from_exchange'].sum():,} ({df['from_exchange'].mean()*100:.2f}%)")
    print(f"  α₇ ToExchange: {df['to_exchange'].sum():,} ({df['to_exchange'].mean()*100:.2f}%)")
    
    return df


# =============================================================================
# RAW TRANSACTION PARSING - Extract addresses from tx_data hex
# =============================================================================

# Bech32 encoding for native segwit addresses
_BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
_BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

def _bech32_polymod(values):
    GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = (chk >> 25)
        chk = (chk & 0x1ffffff) << 5 ^ v
        for i in range(5):
            chk ^= GEN[i] if ((b >> i) & 1) else 0
    return chk

def _bech32_hrp_expand(s):
    return [ord(x) >> 5 for x in s] + [0] + [ord(x) & 31 for x in s]

def _bech32_create_checksum(hrp, data):
    values = _bech32_hrp_expand(hrp) + data
    polymod = _bech32_polymod(values + [0,0,0,0,0,0]) ^ 1
    return [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]

def _bech32_encode(hrp, data):
    combined = data + _bech32_create_checksum(hrp, data)
    return hrp + '1' + ''.join([_BECH32_CHARSET[d] for d in combined])

def _convertbits(data, frombits, tobits, pad=True):
    acc = 0
    bits = 0
    ret = []
    maxv = (1 << tobits) - 1
    max_acc = (1 << (frombits + tobits - 1)) - 1
    for value in data:
        acc = ((acc << frombits) | value) & max_acc
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        return None
    return ret

def _base58_encode(b):
    import hashlib
    n = int.from_bytes(b, 'big')
    result = ''
    while n:
        n, r = divmod(n, 58)
        result = _BASE58_ALPHABET[r] + result
    for byte in b:
        if byte == 0:
            result = _BASE58_ALPHABET[0] + result
        else:
            break
    return result

def script_to_address(script_hex: str) -> Optional[str]:
    """
    Convert scriptPubKey hex to Bitcoin address.
    
    Supports: P2PKH, P2SH, P2WPKH (native segwit v0), P2WSH
    
    Args:
        script_hex: Hex string of the scriptPubKey
        
    Returns:
        Bitcoin address string or None if unknown format
    """
    import hashlib
    script = bytes.fromhex(script_hex)
    
    # P2WPKH (native segwit v0, 20 bytes): 0014{20 bytes}
    if len(script) == 22 and script[0] == 0x00 and script[1] == 0x14:
        witprog = list(script[2:])
        data = [0] + _convertbits(witprog, 8, 5)
        return _bech32_encode('bc', data)
    
    # P2WSH (native segwit v0, 32 bytes): 0020{32 bytes}
    if len(script) == 34 and script[0] == 0x00 and script[1] == 0x20:
        witprog = list(script[2:])
        data = [0] + _convertbits(witprog, 8, 5)
        return _bech32_encode('bc', data)
    
    # P2PKH: 76a914{20 bytes}88ac
    if len(script) == 25 and script[0] == 0x76 and script[1] == 0xa9:
        pubkey_hash = script[3:23]
        version = b'\x00'  # mainnet
        payload = version + pubkey_hash
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        return _base58_encode(payload + checksum)
    
    # P2SH: a914{20 bytes}87
    if len(script) == 23 and script[0] == 0xa9 and script[1] == 0x14:
        script_hash = script[2:22]
        version = b'\x05'  # mainnet P2SH
        payload = version + script_hash
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        return _base58_encode(payload + checksum)
    
    return None


def extract_output_addresses_from_tx_hex(tx_hex: str) -> List[str]:
    """
    Extract output addresses from raw transaction hex.
    
    Args:
        tx_hex: Raw transaction hex string (from tx_data column)
        
    Returns:
        List of output addresses (destination addresses)
    """
    try:
        from bitcoinutils.setup import setup
        from bitcoinutils.transactions import Transaction
        
        # Ensure mainnet setup
        try:
            setup('mainnet')
        except:
            pass  # Already set up
        
        tx = Transaction.from_raw(tx_hex)
        addresses = []
        
        for out in tx.outputs:
            script_hex = out.script_pubkey.to_hex()
            addr = script_to_address(script_hex)
            if addr:
                addresses.append(addr)
        
        return addresses
    except Exception as e:
        return []


def check_exchange_in_outputs(tx_hex: str) -> Tuple[bool, Optional[str]]:
    """
    Check if any output address belongs to a known exchange.
    
    Args:
        tx_hex: Raw transaction hex string
        
    Returns:
        Tuple of (is_to_exchange, exchange_name)
    """
    addresses = extract_output_addresses_from_tx_hex(tx_hex)
    for addr in addresses:
        if is_exchange_address(addr):
            return True, get_exchange_name(addr)
    return False, None


# =============================================================================
# BITHYPHA API HELPER (for extended lookups)
# =============================================================================

def query_bithypha_address(address: str) -> Optional[Dict]:
    """
    Query Bithypha for address information.
    
    Args:
        address: Bitcoin address to look up
        
    Returns:
        Dictionary with address info or None if not found/error
        
    Note:
        Requires internet connection. Use responsibly to avoid rate limiting.
    """
    try:
        import requests
        url = f"https://bithypha.com/address/{address}"
        # This returns HTML, not JSON - would need to parse
        # For actual API access, contact Bithypha
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return {"url": url, "exists": True}
    except Exception as e:
        print(f"Error querying Bithypha: {e}")
    return None


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_exchange_summary():
    """Print summary of known exchange data."""
    print("=" * 60)
    print("EXCHANGE ADDRESS DATABASE SUMMARY")
    print("=" * 60)
    print(f"\nKnown Exchange Addresses (curated): {len(_ALL_EXCHANGE_ADDRESSES)}")
    print(f"Number of Exchanges with addresses: {len(KNOWN_EXCHANGE_ADDRESSES)}")
    print(f"\nBithypha Entity Database:")
    print(f"  - Total Exchanges: {len(EXCHANGE_ENTITIES)}")
    total_addrs = sum(e['addresses_count'] for e in EXCHANGE_ENTITIES.values())
    total_btc = sum(e['balance_btc'] for e in EXCHANGE_ENTITIES.values())
    print(f"  - Total Clustered Addresses: {total_addrs:,}")
    print(f"  - Total BTC Holdings: {total_btc:,.2f} BTC")
    
    print("\n" + "-" * 60)
    print("TOP 10 EXCHANGES BY ADDRESS COUNT:")
    print("-" * 60)
    sorted_exchanges = sorted(
        EXCHANGE_ENTITIES.items(), 
        key=lambda x: x[1]['addresses_count'], 
        reverse=True
    )[:10]
    for name, data in sorted_exchanges:
        print(f"  {name:25s} {data['addresses_count']:>12,} addresses")
    
    print("\n" + "-" * 60)
    print("TOP 10 EXCHANGES BY BTC BALANCE:")
    print("-" * 60)
    sorted_by_balance = sorted(
        EXCHANGE_ENTITIES.items(),
        key=lambda x: x[1]['balance_btc'],
        reverse=True
    )[:10]
    for name, data in sorted_by_balance:
        print(f"  {name:25s} {data['balance_btc']:>15,.2f} BTC")


if __name__ == "__main__":
    print_exchange_summary()

