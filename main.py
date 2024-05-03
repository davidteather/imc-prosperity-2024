import json
from json import JSONEncoder
import numpy as np
from typing import List, Tuple, Dict
import jsonpickle
import math

def cdf(x):
    """Approximation of the cumulative distribution function for the standard normal distribution."""
    # Constants for the approximation
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911

    # Save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2.0)

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * math.exp(-x*x)

    return 0.5 * (1.0 + sign * y)



"""
 Data Model
"""

Time = int
Symbol = str
Product = str
Position = int
UserId = str
Observation = int


class Listing:
    def __init__(self, symbol: Symbol, product: Product, denomination: Product):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination


class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"

    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"


class OrderDepth:
    def __init__(self, buy_orders=None, sell_orders=None):
        if sell_orders is None:
            sell_orders = {}
        if buy_orders is None:
            buy_orders = {}
        self.buy_orders: Dict[int, int] = buy_orders
        self.sell_orders: Dict[int, int] = sell_orders


class Trade:
    def __init__(self, symbol: Symbol, price: int, quantity: int, buyer: UserId = None, seller: UserId = None,
                 timestamp: int = 0) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

class TradingState(object):
    def __init__(self,
                 traderData: str,
                 timestamp: int,
                 listings: Dict[str, Listing],
                 order_depths: Dict[str, OrderDepth],
                 own_trades: Dict[str, List[Trade]],
                 market_trades: Dict[str, List[Trade]],
                 position: Dict[str, int],
                 observations: Observation):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations
        
    def toJSON(self):
       return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)


class ProsperityEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

"""
Strategies
"""

class Strategy:
    def __init__(self, name: str, max_position: int):
        self.name: str = name
        self.cached_prices: list = []
        self.cached_means: list = []
        self.max_pos: int = max_position
        self.trade_count: int = 1

        self.prod_position: int = 0
        self.new_buy_orders: int = 0
        self.new_sell_orders: int = 0
        self.order_depth: OrderDepth = OrderDepth()

    def reset_from_state(self, state: TradingState):
        self.prod_position = state.position[self.name] if self.name in state.position.keys() else 0
        self.order_depth: OrderDepth = state.order_depths[self.name]

        self.new_buy_orders = 0
        self.new_sell_orders = 0

    def sell_product(self, best_bids, i, order_depth, orders):
        # Sell product at best bid
        best_bid_volume = order_depth.buy_orders[best_bids[i]]
        if self.prod_position - best_bid_volume >= -self.max_pos:
            orders.append(Order(self.name, best_bids[i], -best_bid_volume))
            self.prod_position += -best_bid_volume
            self.new_sell_orders += best_bid_volume

        else:
            # Sell as much as we can without exceeding the self.max_pos[product]
            vol = self.prod_position + self.max_pos
            orders.append(Order(self.name, best_bids[i], -vol))
            self.prod_position += -vol
            self.new_sell_orders += vol

    def buy_product(self, best_asks, i, order_depth, orders):
        # Buy product at best ask
        best_ask_volume = order_depth.sell_orders[best_asks[i]]
        if self.prod_position - best_ask_volume <= self.max_pos:
            orders.append(Order(self.name, best_asks[i], -best_ask_volume))
            self.prod_position += -best_ask_volume
            self.new_buy_orders += -best_ask_volume
        else:
            # Buy as much as we can without exceeding the self.max_pos[product]
            vol = self.max_pos - self.prod_position
            orders.append(Order(self.name, best_asks[i], vol))
            self.prod_position += vol
            self.new_buy_orders += vol

    def continuous_buy(self, order_depth: OrderDepth, orders: list):
        if len(order_depth.sell_orders) != 0:
            best_asks = sorted(order_depth.sell_orders.keys())

            i = 0
            while i < self.trade_count and len(best_asks) > i:
                if self.prod_position == self.max_pos:
                    break

                self.buy_product(best_asks, i, order_depth, orders)
                i += 1

    def continuous_sell(self, order_depth: OrderDepth, orders: list):
        if len(order_depth.buy_orders) != 0:
            best_bids = sorted(order_depth.buy_orders.keys(), reverse=True)

            i = 0
            while i < self.trade_count and len(best_bids) > i:
                if self.prod_position == -self.max_pos:
                    break

                self.sell_product(best_bids, i, order_depth, orders)
                i += 1


class CrossStrategy(Strategy):
    def __init__(self, name: str, min_req_price_difference: int, max_position: int):
        super().__init__(name, max_position)
        self.strategy_start_day = 2

        self.old_asks = []
        self.old_bids = []
        self.min_req_price_difference = min_req_price_difference

    def trade(self, trading_state: TradingState, orders: list):
        order_depth: OrderDepth = trading_state.order_depths[self.name]
        self.cache_prices(order_depth)
        if len(self.old_asks) < self.strategy_start_day or len(self.old_bids) < self.strategy_start_day:
            return

        avg_bid, avg_ask = self.calculate_prices(self.strategy_start_day)

        if len(order_depth.sell_orders) != 0:
            best_asks = sorted(order_depth.sell_orders.keys())

            i = 0
            while i < self.trade_count and len(best_asks) > i and best_asks[i] - avg_bid <= self.min_req_price_difference:
                if self.prod_position == self.max_pos:
                    break
                self.buy_product(best_asks, i, order_depth, orders)
                i += 1

        if len(order_depth.buy_orders) != 0:
            best_bids = sorted(order_depth.buy_orders.keys(), reverse=True)

            i = 0
            while i < self.trade_count and len(best_bids) > i and avg_ask - best_bids[i] <= self.min_req_price_difference:
                if self.prod_position == -self.max_pos:
                    break
                self.sell_product(best_bids, i, order_depth, orders)

                i += 1

    def calculate_prices(self, days: int) -> Tuple[int, int]:
        # Calculate the average bid and ask price for the last days

        relevant_bids = []
        for bids in self.old_bids[-days:]:
            relevant_bids.extend([(value, bids[value]) for value in bids])
        relevant_asks = []
        for asks in self.old_asks[-days:]:
            relevant_asks.extend([(value, asks[value]) for value in asks])

        avg_bid = np.average([x[0] for x in relevant_bids], weights=[x[1] for x in relevant_bids])
        avg_ask = np.average([x[0] for x in relevant_asks], weights=[x[1] for x in relevant_asks])

        return avg_bid, avg_ask

    def cache_prices(self, order_depth: OrderDepth):
        sell_orders = order_depth.sell_orders
        buy_orders = order_depth.buy_orders

        self.old_asks.append(sell_orders)
        self.old_bids.append(buy_orders)


class DiffStrategy(Strategy):
    def __init__(self, name: str, max_pos: int, derivative_resolution: int, diff_thresh: int):
        super().__init__(name, max_pos)
        self.derivative_resolution: int = derivative_resolution
        self.diff_thresh: int = diff_thresh

    def trade(self, trading_state: TradingState, orders: list):
        order_depth: OrderDepth = trading_state.order_depths[self.name]
        self.cache_purchased_prices(trading_state)
        self.calculate_means()

        diff = self.get_price_difference()

        if diff < -self.diff_thresh and len(order_depth.sell_orders) != 0:
            self.continuous_buy(order_depth, orders)

        if diff > self.diff_thresh and len(order_depth.buy_orders) != 0:
            self.continuous_sell(order_depth, orders)

    def get_price_difference(self) -> float:
        # Calculate the difference between the current mean and the mean from
        # self.derivative_resolution days ago
        if len(self.cached_means) < self.derivative_resolution + 1:
            old_mean = self.cached_means[0]
        else:
            old_mean = self.cached_means[-self.derivative_resolution]
        diff = self.cached_means[-1] - old_mean
        return diff

    def calculate_means(self):
        #
        if len(self.cached_prices) == 0:
            self.cached_means.append(0)

        else:
            relevant_prices = []
            for day_prices in self.cached_prices[max(-len(self.cached_prices), -1):]:
                for price in day_prices:
                    relevant_prices.append(price)
            prices = np.array([x[1] for x in relevant_prices])
            quantities = np.abs(np.array([x[0] for x in relevant_prices]))

            self.cached_means.append(np.average(prices, weights=quantities))

    def cache_purchased_prices(self, state: TradingState) -> None:
        # Caches prices of bought and sold products

        market_trades = state.market_trades
        own_trades = state.own_trades

        prod_trades: List[Trade] = own_trades.get(self.name, []) + market_trades.get(self.name, [])

        if len(prod_trades) > 0:
            prices = [(trade.quantity, trade.price) for trade in prod_trades]
            self.cached_prices.append(prices)


class NullStrategy(Strategy):
    def __init__(self, name: str, max_pos: int, **kwargs):
        super().__init__(name, max_pos)
    
    def trade(self, trading_state: TradingState, orders: list):
        pass

class FixedStrategy(Strategy):
    def __init__(self, name: str, max_pos: int):
        super().__init__(name, max_pos)
        self.amethyst_price = 10000
        self.amethyst_diff = 4

    def trade(self, trading_state: TradingState, orders: list):
        order_depth: OrderDepth = trading_state.order_depths[self.name]
        # Check if there are any SELL orders
        if len(order_depth.sell_orders) > 0:
            #
            # self.cache_prices(order_depth)
            # Sort all the available sell orders by their price
            best_asks = sorted(order_depth.sell_orders.keys())

            # Check if the lowest ask (sell order) is lower than the above defined fair value
            i = 0
            while i < self.trade_count and best_asks[i] < self.amethyst_price:
                # Fill ith ask order if it's below the acceptable
                if self.prod_position == self.max_pos:
                    break
                self.buy_product(best_asks, i, order_depth, orders)
                i += 1
        if len(order_depth.buy_orders) != 0:
            best_bids = sorted(order_depth.buy_orders.keys(), reverse=True)

            i = 0
            while i < self.trade_count and best_bids[i] > self.amethyst_price:
                if self.prod_position == -self.max_pos:
                    break
                self.sell_product(best_bids, i, order_depth, orders)
                i += 1


class ObservationStrategy(Strategy):
    def __init__(self, name: str, max_position: int):
        super().__init__(name, max_position)
        self.old_dolphins = -1
        self.dolphins_spotted = False
        self.dolphins_gone = False

        self.dolphin_action_time = 900
        self.gear_timestamp_diff = 70000

        self.dolphins_spotted_timestamp = -1
        self.dolphins_gone_timestamp = -1

    def handle_observations(self, trading_state: TradingState):
        for observation in trading_state.observations.keys():
            if observation == "DOLPHIN_SIGHTINGS":
                if self.old_dolphins == -1:
                    self.old_dolphins = trading_state.observations["DOLPHIN_SIGHTINGS"]
                    continue
                if trading_state.observations["DOLPHIN_SIGHTINGS"] - self.old_dolphins > 10:
                    # print("DOLPHINS SPOTTED")
                    self.dolphins_spotted = True
                    self.dolphins_spotted_timestamp = trading_state.timestamp
                if trading_state.observations["DOLPHIN_SIGHTINGS"] - self.old_dolphins < -10:
                    # print("DOLPHINS GONE")
                    self.dolphins_gone = True
                    self.dolphins_gone_timestamp = trading_state.timestamp
                self.old_dolphins = trading_state.observations["DOLPHIN_SIGHTINGS"]

    def trade(self, trading_state: TradingState, orders: list):

        self.handle_observations(trading_state)

        order_depth: OrderDepth = trading_state.order_depths[self.name]

        if self.dolphins_spotted and trading_state.timestamp - self.dolphins_spotted_timestamp < self.dolphin_action_time:
            # start buying gear if dolphins have been spotted
            #print(self.dolphins_spotted_timestamp)
            # print("BUYING GEAR")
            #print(trading_state.timestamp)
            self.continuous_buy(order_depth, orders)

        if self.dolphins_gone and trading_state.timestamp - self.dolphins_gone_timestamp < self.dolphin_action_time:
            # start selling gear if dolphins are going away
            # print("SELLING GEAR")
            self.continuous_sell(order_depth, orders)
        if self.dolphins_spotted and trading_state.timestamp - self.dolphins_spotted_timestamp > self.gear_timestamp_diff:
            # Start selling after dolphins have been spotted for long enough
            self.continuous_sell(order_depth, orders)

            if trading_state.timestamp - self.dolphins_spotted_timestamp - self.gear_timestamp_diff > self.dolphin_action_time:
                self.dolphins_spotted = False

        if self.dolphins_gone and trading_state.timestamp - self.dolphins_gone_timestamp > self.gear_timestamp_diff:
            # Start buying after dolphins have been gone for long enough
            self.continuous_buy(order_depth, orders)
            if trading_state.timestamp - self.dolphins_gone_timestamp - self.gear_timestamp_diff > self.dolphin_action_time:
                self.dolphins_gone = False

class TimeBasedStrategy(CrossStrategy):
    def __init__(self, name, min_req_price_difference, max_position):
        super().__init__(name, min_req_price_difference, max_position)
        self.berries_ripe_timestamp = 350000
        self.berries_peak_timestamp = 500000
        self.berries_sour_timestamp = 650000

    def trade(self, trading_state: TradingState, orders: list):
        order_depth = trading_state.order_depths[self.name]
        if 0 < trading_state.timestamp - self.berries_ripe_timestamp < 5000:
            # print("BERRIES ALMOST RIPE")

            # start buying berries if they start being ripe
            if len(order_depth.sell_orders) != 0:
                best_asks = sorted(order_depth.sell_orders.keys())

                i = 0
                while i < self.trade_count and len(best_asks) > i:
                    if self.prod_position == -self.max_pos:
                        break
                    self.buy_product(best_asks, i, order_depth, orders)
                    i += 1

        elif 0 < trading_state.timestamp - self.berries_peak_timestamp < 5000:
            # print("BERRIES READY TO SELL")
            self.continuous_sell(order_depth, orders)
        else:
            super().trade(trading_state, orders)


"""
Products
"""
# 2024 Products

# Tutorial Products
class Amethysts(FixedStrategy):
    def __init__(self):
        super().__init__("AMETHYSTS", max_pos=20)

class Starfruit(CrossStrategy):
    def __init__(self):
        super().__init__("STARFRUIT", min_req_price_difference=3, max_position=20)


"""
2023 Products
class Pearls(FixedStrategy):
    def __init__(self):
        super().__init__("PEARLS", max_pos=20)


class Bananas(CrossStrategy):
    def __init__(self):
        super().__init__("BANANAS", min_req_price_difference=3, max_position=20)


class PinaColadas(DiffStrategy):
    def __init__(self):
        super().__init__("PINA_COLADAS", max_position=150, derivative_resolution=150, diff_thresh=30)


class Baguette(DiffStrategy):
    def __init__(self):
        super().__init__("BAGUETTE", max_position=150, derivative_resolution=20, diff_thresh=200)


class Dip(DiffStrategy):
    def __init__(self):
        super().__init__("DIP", max_position=300, derivative_resolution=100, diff_thresh=40)


class Coconut(DiffStrategy):
    def __init__(self):
        super().__init__("COCONUTS", max_position=600, derivative_resolution=1500, diff_thresh=30)


class Ukulele(DiffStrategy):
    def __init__(self):
        super().__init__("UKULELE", max_position=70, derivative_resolution=20, diff_thresh=150)


class Basket(DiffStrategy):
    def __init__(self):
        super().__init__("PICNIC_BASKET", max_position=70, derivative_resolution=50, diff_thresh=100)


class Berries(TimeBasedStrategy):
    def __init__(self):
        super().__init__("BERRIES", min_req_price_difference=2, max_position=250)


class DivingGear(DiffStrategy):
    def __init__(self):
        super().__init__("DIVING_GEAR", max_position=50, derivative_resolution=15, diff_thresh=25)
"""

"""
Conversin
"""
class ConversionObservation:

    def __init__(self, bidPrice: float, askPrice: float, transportFees: float, exportTariff: float, importTariff: float, sunlight: float, humidity: float):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sunlight = sunlight
        self.humidity = humidity

class Observation:

    def __init__(self, plainValueObservations: Dict[Product, int], conversionObservations: Dict[Product, ConversionObservation]) -> None:
        self.plainValueObservations = plainValueObservations
        self.conversionObservations = conversionObservations
        
    def __str__(self) -> str:
        return "(plainValueObservations: " + jsonpickle.encode(self.plainValueObservations) + ", conversionObservations: " + jsonpickle.encode(self.conversionObservations) + ")"


class OrchidStrategy(Strategy):
    """
    * import only
    * delicate and can't be vlaued right
    * depend on sunlight and humdiity

    * if sunlight exposure less than 7 hours a day
    * production decrease with 4% for every 10 minutes
    * ideal humitiy is 60-80%
    * outside of this production falls 2% for every 5% point of humidity change
    * has shipping costs, in and export tariffs
    * last we need storage space to keep the storage when they're here
    * orchids over this can't
    * good storage is .1 seashell per orchid per timestamp
    * seemed like by the UI the storage is max 5K units?
    """

    def __init__(self, name: str, max_position: int, storage_cost_per_unit: float):
        super().__init__(name, max_position)
        self.storage_cost_per_unit = storage_cost_per_unit  # Cost to store each orchid per timestamp

    def trade(self, trading_state: TradingState, orders: list):
        # Fetch current environmental observations
        obs = trading_state.observations.conversionObservations[self.name]
        sunlight = obs.sunlight  # hours
        """
        note: I think the most continuous question everyone is having is the units for
        sunlight. Tropical TV says that something happens when sunlight is
        "under 7 hours a day", but the units for sunlight are in huge numbers,
        such as 2500. How do we convert those to hours a day? Do we divide by 365? 

        """
        humidity = obs.humidity  # percentage
        avg_bid = obs.bidPrice  # average bid price
        avg_ask = obs.askPrice  # average ask price

        # Calculate production impacts
        price_adjustment = self.calculate_price_adjustment(sunlight, humidity)

        # Use average ask price as a base for the expected price, adjust based on conditions
        expected_price = avg_ask + price_adjustment - self.storage_cost_per_unit * self.prod_position

        # Add logistic costs to the expected price
        expected_price += self.calculate_logistics_cost(obs)

        # Implement buying or selling logic
        order_depth = trading_state.order_depths[self.name]
        self.execute_trading_logic(order_depth, orders, expected_price)

    def calculate_price_adjustment(self, sunlight, humidity):
        adjustment = 0
        # Calculate sunlight effect
        if sunlight < 7:
            missing_minutes = (7 - sunlight) * 60
            adjustment -= missing_minutes / 10 * 0.04  # 4% decrease per 10 minutes below 7 hours

        # Calculate humidity effect
        if humidity < 60:
            adjustment -= (60 - humidity) / 5 * 0.02
        elif humidity > 80:
            adjustment -= (humidity - 80) / 5 * 0.02

        return adjustment

    def calculate_logistics_cost(self, obs: ConversionObservation):
        return obs.transportFees + obs.exportTariff + obs.importTariff

    def execute_trading_logic(self, order_depth, orders, expected_price):
        best_asks = sorted(order_depth.sell_orders.keys())
        best_bids = sorted(order_depth.buy_orders.keys(), reverse=True)

        # Decision to buy
        for ask_price in best_asks:
            if ask_price < expected_price and self.prod_position < self.max_pos:
                quantity = min(order_depth.sell_orders[ask_price], self.max_pos - self.prod_position)
                orders.append(Order(self.name, ask_price, quantity))
                self.prod_position += quantity
                break  # Buy only at the best price

        # Decision to sell
        for bid_price in best_bids:
            if bid_price > expected_price and self.prod_position > 0:
                quantity = min(order_depth.buy_orders[bid_price], self.prod_position)
                orders.append(Order(self.name, bid_price, -quantity))
                self.prod_position -= quantity
                break


class Orchids(OrchidStrategy):
    def __init__(self):
        super().__init__("ORCHIDS", max_position=100, storage_cost_per_unit=0.1)

class Chocolates(NullStrategy):
    def __init__(self):
        super().__init__("CHOCOLATE", max_pos=250, derivative_resolution=1500, diff_thresh=200)

class Strawberries(NullStrategy):
    def __init__(self):
        super().__init__("STRAWBERRIES", max_pos=350, derivative_resolution=1500, diff_thresh=200)


class Roses(DiffStrategy):
    def __init__(self):
        super().__init__("ROSES", max_pos=60, derivative_resolution=500, diff_thresh=150)

class BasketStrategy(Strategy):
    def __init__(self, name: str, max_pos: int, premium: int):
        super().__init__(name, max_pos)
        self.premium = premium

    def calculate_expected_price(self, chocolate_price, strawberries_price, roses_price):
        # Calculate the expected price of the gift basket based on its components
        return (4 * chocolate_price) + (6 * strawberries_price) + roses_price + self.premium

    def trade(self, trading_state: TradingState, orders: list):
        # Get current market prices of the components and the basket
        chocolate_price = self.get_market_price(trading_state, "CHOCOLATE")
        strawberries_price = self.get_market_price(trading_state, "STRAWBERRIES")
        roses_price = self.get_market_price(trading_state, "ROSES")
        gift_basket_price = self.get_market_price(trading_state, "GIFT_BASKET")

        # Calculate the expected market price of the gift basket
        expected_price = self.calculate_expected_price(chocolate_price, strawberries_price, roses_price)

        # Determine the action based on the comparison of current and expected prices
        if gift_basket_price > expected_price:
            # Sort orders to sell Gift Basket
            self.continuous_sell(trading_state.order_depths["GIFT_BASKET"], orders)

        elif gift_basket_price < expected_price:
            # Sort orders to buy Gift Basket
            self.continuous_buy(trading_state.order_depths["GIFT_BASKET"], orders)

    def get_market_price(self, trading_state, product):
        # Simplified market price retrieval from order depth
        order_depth = trading_state.order_depths[product]
        if order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        return float('inf')  # Return a high price if no sell orders are available

    def buy_components(self, trading_state, orders, chocolate_price, strawberries_price, roses_price):
        # Buy components if there's enough room in position limits
        self.continuous_buy_component("CHOCOLATE", chocolate_price, 4, trading_state, orders)
        self.continuous_buy_component("STRAWBERRIES", strawberries_price, 6, trading_state, orders)
        self.continuous_buy_component("ROSES", roses_price, 1, trading_state, orders)

    def sell_components(self, trading_state, orders, chocolate_price, strawberries_price, roses_price):
        # Sell components
        self.continuous_sell_component("CHOCOLATE", chocolate_price, 4, trading_state, orders)
        self.continuous_sell_component("STRAWBERRIES", strawberries_price, 6, trading_state, orders)
        self.continuous_sell_component("ROSES", roses_price, 1, trading_state, orders)

    def continuous_buy_component(self, component_name, price, quantity, trading_state, orders):
        if trading_state.position.get(component_name, 0) + quantity <= self.max_pos:
            orders.append(Order(component_name, price, quantity))

    def continuous_sell_component(self, component_name, price, quantity, trading_state, orders):
        if trading_state.position.get(component_name, 0) >= quantity:
            orders.append(Order(component_name, price, -quantity))

class GiftBaskets(BasketStrategy):
    def __init__(self):
        super().__init__("GIFT_BASKET", max_pos=60, premium=375)


class BlackScholesStrategy(Strategy):
    def __init__(self, name: str, strike_price: float, maturity: int, max_pos: int):
        super().__init__(name, max_pos)
        self.strike_price = strike_price
        self.maturity = maturity  # days until expiration
        self.r = 0.0  # risk-free rate, assume 0 for simplicity
        self.sigma = 0.2  # initial volatility estimate, adjust as needed

    def update_volatility(self, historical_prices):
        # Use Exponentially Weighted Moving Average (EWMA) to update volatility
        lambda_ = 0.94  # Decay factor for EWMA, common choice in finance
        if len(historical_prices) > 1:
            returns = np.diff(historical_prices) / historical_prices[:-1]
            var = np.var(returns)
            if hasattr(self, 'sigma'):
                self.sigma = np.sqrt(lambda_ * self.sigma**2 + (1 - lambda_) * var) * np.sqrt(252)
            else:
                self.sigma = np.sqrt(var) * np.sqrt(252)

    def black_scholes_price(self, current_price, time_to_maturity, premium = 0):
        if current_price <= 0 or self.strike_price <= 0 or time_to_maturity <= 0 or self.sigma <= 0:
            return 0
        S = current_price
        K = self.strike_price
        T = time_to_maturity / 365
        r = self.r
        sigma = self.sigma
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = (S * cdf(d1) - (K + premium) * np.exp(-r * T) * cdf(d2))
        
        return call_price

    def trade(self, trading_state: TradingState, orders: list):
        current_price = self.get_current_price(trading_state, self.name.replace('_COUPON', ''))
        self.update_volatility(self.cached_prices)
        premium = self.calculate_dynamic_premium(self.maturity)  # Calculate premium dynamically
        theoretical_price = self.black_scholes_price(current_price, self.maturity, premium)
        order_depth = trading_state.order_depths[self.name]
        self.execute_trading_logic(order_depth, orders, theoretical_price, current_price)

    def calculate_dynamic_premium(self, days_until_expiration):
        # Dynamic premium calculation based on remaining days and volatility
        base_premium = 637.63  # Base premium, adjust as needed
        return base_premium
        return base_premium * (1 + self.sigma / 100) * (days_until_expiration / self.maturity)

    def execute_trading_logic(self, order_depth, orders, theoretical_price, current_price):
        # Sort the asks and bids so that we can access the best prices
        best_asks = sorted(order_depth.sell_orders.keys())
        best_bids = sorted(order_depth.buy_orders.keys(), reverse=True)

        # Go through the asks and determine if we should buy
        for i, ask_price in enumerate(best_asks):
            if ask_price < theoretical_price:
                self.buy_product(best_asks, i, order_depth, orders)

        # Go through the bids and determine if we should sell
        for i, bid_price in enumerate(best_bids):
            if bid_price > theoretical_price:
                self.sell_product(best_bids, i, order_depth, orders)

    def get_current_price(self, trading_state, product):
        # Retrieve the current price from order depth
        order_depth = trading_state.order_depths[product]
        if order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        return float('inf')  # High price if no sell orders are availabl

class Coconuts(NullStrategy):
    def __init__(self):
        super().__init__("COCONUT", max_pos=300)

class CoconutCoupons(NullStrategy):
    def __init__(self):
        round = 4
        super().__init__("COCONUT_COUPON", max_pos=600, strike_price=10000, maturity=250 - round)

"""
Logger
"""
import json
from typing import Any

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

"""
    Trader Logic
"""
AMETHYST_PRICE = 10000

class Trader:
    def __init__(self):
        self.products = {
            "AMETHYSTS": Amethysts(),
            "STARFRUIT": Starfruit(),
            "ORCHIDS": Orchids(),
            "CHOCOLATE": Chocolates(),
            "STRAWBERRIES": Strawberries(),
            "ROSES": Roses(),
            "GIFT_BASKET": GiftBaskets(),
            "COCONUT": Coconuts(),
            "COCONUT_COUPON": CoconutCoupons()
        }
        self.acceptable_risk_threshold = 0.1

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
            if product in self.products.keys():
                orders: list[Order] = []

                self.products[product].reset_from_state(state)
                self.products[product].trade(trading_state=state, orders=orders)
                result[product] = orders

        conversions = None
        # conversion_opportunity, conversions = self.analyze_conversion_opportunity(state)
        
        logger.flush(state, result, conversions, state.traderData)
        return result, conversions, state.traderData
    
    """
    Conversion stuff that's mostly chatgpt
    Currently not used and our backtester here doesn't support it
    Maybe switch to https://github.com/jmerle/imc-prosperity-2-backtester
    Can run: prosperity2bt main.py 0 --vis

    """

    def analyze_conversion_opportunity(self, state: TradingState):
        conversion_opportunity = False
        conversions = 0

        # Accessing conversionObservations directly from the provided state.observations
        if "STARFRUIT" in state.observations.conversionObservations and "AMETHYSTS" in state.position:
            conversion_info = state.observations.conversionObservations["STARFRUIT"]
            amethyst_position = state.position["AMETHYSTS"]
            
            expected_future_price = self.predict_future_price("STARFRUIT", state)
            current_ask_price = conversion_info.askPrice
            conversion_cost = conversion_info.transportFees + conversion_info.exportTariff + conversion_info.importTariff
            
            potential_profit_per_unit = expected_future_price - current_ask_price - conversion_cost
            
            risk_factor = self.calculate_risk_factor("STARFRUIT", state)
            
            if potential_profit_per_unit > 0 and risk_factor < self.acceptable_risk_threshold:
                conversion_opportunity = True
                confidence_level = self.calculate_confidence_level("STARFRUIT", state)
                max_conversion_limit = min(amethyst_position, 20)  # Assuming a position limit for STARFRUIT
                conversions = max_conversion_limit * confidence_level
            
        return conversion_opportunity, int(conversions)

    def predict_future_price(self, product, state):
        # Implement predictive modeling to forecast future price
        # For simplicity, using moving average as a placeholder
        moving_average = self.calculate_moving_average(product, state)
        volatility = self.calculate_volatility(product, state)
        trend_direction = self.identify_trend_direction(product, state)
        return moving_average * (1 + trend_direction * volatility)

    def calculate_risk_factor(self, product, state):
        # Simplified risk calculation based on market volatility
        volatility = self.calculate_volatility(product, state)
        return volatility / self.calculate_average_volatility(state)

    def calculate_confidence_level(self, product, state):
        # Placeholder for confidence level calculation
        # Could be based on model accuracy, backtesting, etc.
        return 0.5  # Example static value, should be dynamically calculated