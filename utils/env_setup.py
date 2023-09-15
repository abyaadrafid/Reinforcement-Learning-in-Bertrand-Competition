from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


def get_stock_env(data):
    action_space = len(data.tic.unique())
    state_space = 1 + 2 * action_space + len(INDICATORS) * action_space
    buy_cost_list = sell_cost_list = [0.001] * action_space
    num_stock_shares = [0] * action_space

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": action_space,
        "tech_indicator_list": INDICATORS,
        "action_space": action_space,
        "reward_scaling": 1e-4,
    }

    env, _ = StockTradingEnv(df=data, **env_kwargs).get_sb_env()

    return env
