from utils.env_setup import get_stock_env
from utils.preprocess import process_dataset

if __name__ == "__main__":
    train, valid = process_dataset()
    env = get_stock_env(train)
    print(env.reset())
