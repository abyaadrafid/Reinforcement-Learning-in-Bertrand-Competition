"""This module is used to make simple Bertrand games"""

import random

import matplotlib.pyplot as plt
import numpy as np


class DemandFunction:
    """Class that generates a demand function"""

    def __init__(self, max_price: int, min_price: int, num_customer=100):
        self.num_customer = num_customer
        self.generate_unform(max_price, min_price)

    def generate_unform(self, max_price, min_price):
        """Deterministic demand function"""
        consumer_prices = np.random.uniform(min_price, max_price, self.num_customer)
        self.demand = consumer_prices

    def get_demand(self):
        """Returns demand function"""
        return self.demand

    def set_demand(self, new_demand):
        """Sets demand"""
        self.demand = new_demand

    def plot_demand(self, show=True):
        """Cummulates demand function and plots"""

        # Preparing
        max_price = int(np.max(self.demand))
        min_price = int(np.min(self.demand))

        cummulative_demand = np.zeros(max_price - min_price)

        # Aggregating
        i = 0
        for price in range(min_price, max_price):

            cummulative_demand[i] = sum(price <= bid for bid in self.demand)
            i += 1

        # Plotting
        plt.plot(cummulative_demand, range(min_price, max_price), label="Demand")
        plt.xlabel("Q")
        plt.ylabel("P")
        plt.title("Demand")

        if show:
            plt.show()


class Market:
    """Class that is supposed to match consumers with sellers"""

    def __init__(
        self,
        num_seller: int,
        num_customer: int,
        max_capacity: int,
        max_price: int,
        min_price: int,
    ):
        # To make it more flexible, read from config send capacities as a dict

        self.demand = DemandFunction(max_price, min_price, num_customer)
        self.num_seller = num_seller
        self.max_capacity = max_capacity
        self.max_revenue = max_capacity * max_price
        self.sellers = [
            Seller(
                name="agent" + str(i),
                capacity=self.max_capacity,
            )
            for i in range(self.num_seller)
        ]

    def add_seller(self, seller):
        """Adds seller to the market"""
        self.sellers.append(seller)
        self.num_seller += 1

    def set_demand(self, demand):
        """Sets demand function"""
        self.demand = demand

    def get_demand(self):
        """Returns demand object"""
        return self.demand

    def plot_market(self):
        """Plots demand function with seller prices"""

        # Plotting demand
        self.demand.plot_demand(False)

        no_of_consumers = len(self.get_demand().get_demand())

        # Plotting seller prices
        for seller in self.sellers:
            plt.plot((0, no_of_consumers), (seller.get_price(), seller.get_price()))
            plt.text(1, seller.get_price() + 1, seller.get_name())

        plt.title("seller Market")
        plt.legend()
        plt.show()

    def allocate_items(self, prices: list[int], render: bool = False):
        # Set seller prices
        # Storing last price if needed
        for index, price in enumerate(prices):
            self.sellers[index].set_price(price)

        # Restock
        for seller in self.sellers:
            seller.restock_inventory()
        # Items are sold
        temp_demand = np.sort(self.demand.get_demand())

        # Consumer decides which seller to buy from
        for consumer_prices in temp_demand:
            checked_sellers = set()

            while len(checked_sellers) < self.num_seller:
                current_seller = random.choice(self.sellers)
                if (
                    current_seller not in checked_sellers
                    and current_seller.get_items_left() > 0
                    and current_seller.get_price() <= consumer_prices
                ):
                    current_seller.sell_item()
                    break
                else:
                    checked_sellers.add(current_seller)

        all_revenue = np.zeros(
            self.num_seller,
        )
        for i, seller in enumerate(self.sellers):
            revenue = (
                seller.get_capacity() - seller.get_items_left()
            ) * seller.get_price()
            all_revenue[i] = revenue

        if render:
            self.print_info()

        # Normalize revenue between -1 and +1, this will be the reward
        for i, revenue in enumerate(all_revenue):
            all_revenue[i] = (revenue / self.max_revenue) * 2 - 1
        return all_revenue

    def print_info(self):

        print("--------------------")
        print("No of consumers: ", len(self.demand.get_demand()), "\n")

        print(
            "{:<15s} {:>13s} {:>13s} {:>13s} {:>13s}".format(
                "seller", "Capacity", "Items left", "Item price", "Revenue"
            )
        )

        for seller in self.sellers:
            revenue = (
                seller.get_capacity() - seller.get_items_left()
            ) * seller.get_price()

            print(
                "{:<15s} {:>13d} {:>13d} {:>13d} {:>13}".format(
                    seller.get_name(),
                    seller.get_capacity(),
                    seller.get_items_left(),
                    seller.get_price(),
                    revenue,
                )
            )

        print("--------------------")


class Seller:
    """Class for Seller. Getter and setter for capacity and price"""

    def __init__(self, name="seller", capacity=0, price=0):
        self.name = name
        self.capacity = capacity
        self.price = price
        self.items_left = capacity

    def set_capacity(self, new_capacity):
        """Sets capacity"""
        self.capacity = new_capacity
        self.items_left = new_capacity

    def set_price(self, new_price):
        """Sets price"""
        self.price = new_price

    def get_capacity(self):
        """Returns capacity"""
        return self.capacity

    def get_price(self):
        """Returns price"""
        return self.price

    def get_items_left(self):
        """Returns remaining items"""
        return self.items_left

    def sell_item(self):
        """Reduces number of availables items by 1"""
        self.items_left -= 1

    def restock_inventory(self):
        """Restocks Items: the number of items left is set equal to capacity"""
        self.items_left = self.capacity

    def get_name(self):
        """Returns name of airline"""
        return self.name

    def set_name(self, new_name):
        """Sets name of airline"""
        self.name = new_name


if __name__ == "__main__":
    market = Market(2, 500, 200, 900, 500)
    for seller in market.sellers:
        print(seller.get_price())
    print(market.allocate_items([900, 0], render=True))
