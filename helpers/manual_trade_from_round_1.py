import numpy as np
from tqdm import tqdm

# Constants
num_simulations = 100000
sell_price = 1000  # Selling price in SeaShells
min_price = 900  # Minimum reserve price
max_price = 1000  # Maximum reserve price
profit_margin = sell_price - min_price

def simulate_bids_linear_distribution():
    best_profit = 0
    best_bid_combo = (0, 0)
    
    for bid1 in tqdm(range(101)):
        for bid2 in range(bid1+1, 101):  # Ensure bid2 is always higher than bid1
            profits = []
            for _ in range(num_simulations):
                # Generate a reserve price with linearly increasing probability
                reserve_price_continuous = np.random.triangular(left=0.00001, mode=100, right=100)
                reserve_price = np.ceil(reserve_price_continuous)
                
                profit = 0

                # Calculate profit for the first bid
                if bid1 > reserve_price:
                    profit += (profit_margin - (bid1 * 10))  # Convert bid back to SeaShells scale
                
                # Calculate profit for the second bid
                elif bid2 > reserve_price:
                    profit += (profit_margin - (bid2 * 10))  # Convert bid back to SeaShells scale

                profits.append(profit)
            
            # Average profit for this bid combination
            avg_profit = np.mean(profits)
            if avg_profit > best_profit:
                best_profit = avg_profit
                best_bid_combo = (bid1, bid2)
                
    return best_bid_combo, best_profit

# Run the simulation with linear distribution
best_bids_linear, max_profit_linear = simulate_bids_linear_distribution()
best_bids_linear_scaled = (best_bids_linear[0] * 10 + 900, best_bids_linear[1] * 10 + 900)  # Scale back to SeaShells
print(best_bids_linear_scaled, max_profit_linear)
