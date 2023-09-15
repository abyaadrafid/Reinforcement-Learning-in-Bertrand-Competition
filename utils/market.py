"""This module is used to make simple Bertrand games"""

import matplotlib.pyplot as plt
import numpy as np


class DemandFunction:
    """Class that generates a demand function"""

    def __init__(self, N=100):
        self.no_consumers = N
        self.demand = np.array([])

    def generate_uniform(self, min_price, max_price):
        """Generates uniformly distributed demand function"""
        self.demand = np.random.uniform(min_price, max_price, self.no_consumers)

    def generate_normal(self, mean, standard_deviation):
        """Generates normal distributed demand function"""
        self.demand = np.random.normal(mean, standard_deviation, self.no_consumers)

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

    def __init__(self):
        self.seller = []
        self.demand = DemandFunction()
        self.no_of_sellers = 0

    def add_seller(self, seller):
        """Adds seller to the market"""
        self.seller.append(seller)
        self.no_of_sellers += 1

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
        for seller in self.seller:
            plt.plot((0, no_of_consumers), (seller.get_price(), seller.get_price()))
            plt.text(1, seller.get_price() + 1, seller.get_name())

        plt.title("seller Market")
        plt.legend()
        plt.show()

    def allocate_items(self):
        """Market clearing function"""

        # Resetting sellers
        for seller in self.seller:
            seller.empty_inventory()

        """
        for seller in self.seller:
            print(seller.get_price())
        """

        # Sorting seller prices by bubble sort
        swapped = True
        while swapped:
            swapped = False
            for i in range(self.no_of_sellers - 1):
                if self.seller[i].get_price() > self.seller[i + 1].get_price():
                    temp1 = self.seller[i]
                    temp2 = self.seller[i + 1]
                    self.seller[i] = temp2
                    self.seller[i + 1] = temp1
                    swapped = True

        # Items are sold
        temp_demand = np.sort(self.demand.get_demand())

        for seller in self.seller:
            i = 0
            for consumer_price in temp_demand:

                if seller.get_items_left() > 0:

                    if seller.get_price() < consumer_price:
                        seller.sell_item()

                    i += 1

            temp_demand = temp_demand[i:-1]

        print("--------------------")
        print("No of consumers: ", len(self.demand.get_demand()), "\n")

        print(
            "{:<15s} {:>13s} {:>13s} {:>13s} {:>13s}".format(
                "seller", "Capacity", "Items left", "Item price", "Revenue"
            )
        )

        for seller in self.seller:
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

    def empty_inventory(self):
        """Empties plane: the number of items left is set equal to capacity"""
        self.items_left = self.capacity

    def get_name(self):
        """Returns name of airline"""
        return self.name

    def set_name(self, new_name):
        """Sets name of airline"""
        self.name = new_name
