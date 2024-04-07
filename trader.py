from typing import Dict, List
from datamodel import Order, TradingState

from products.product import Amethysts, Starfruit
import json

AMETHYST_PRICE = 10000 # verify

class Trader:

    def __init__(self):
        self.products = {
            "AMETHYSTS": Amethysts(),
            "STARFRUIT": Starfruit(),
        }

    def run(self, state: TradingState):
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """

        # Initialize the method output dict as an empty dict
        result = {}

        #print(state)
        #print("traderData: " + state.traderData)
        #print("Observations: " + str(state.observations))
        for product in state.order_depths.keys():
            #print("Product: ", product)
            if product in self.products.keys():
                orders: list[Order] = []

                self.products[product].reset_from_state(state)
                self.products[product].trade(trading_state=state, orders=orders)
                result[product] = orders

        # Sample conversion request. Check more details below in the programming
        conversions = 1

        # Serialize the trader state to JSON or any format that suits your need
        traderData = json.dumps({
            # Include any trader state data you want to preserve across executions
            "someStateData": "SAMPLE"
        })

        return result, conversions, traderData
