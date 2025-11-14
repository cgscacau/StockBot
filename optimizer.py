from itertools import product

def optimize_parameters(df, param_grid, base_cfg: StrategyConfig):
    """
    param_grid: dict como
      {
        "fast_ema": [8, 9, 10],
        "slow_ema": [20, 21, 22],
        "rsi_buy": [25, 30],
        "rsi_sell": [70, 75]
      }
    """
    keys = list(param_grid.keys())
    best_result = None
    best_cfg = None

    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        cfg = StrategyConfig(**{**base_cfg.__dict__, **params})
        df_signals = generate_signals(df, cfg)
        res = backtest(df_signals)
        metric = res["sharpe"]  # ou outro critério

        if (best_result is None) or (metric > best_result):
            best_result = metric
            best_cfg = cfg

    return best_cfg, best_result
