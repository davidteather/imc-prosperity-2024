class TileAttributes:
    def __init__(self, row, col, multiplier, hunters):
        self.row = row
        self.col = col
        self.multiplier = multiplier
        self.hunters = hunters

    def real_reward(self, base_reward, percentage_of_pop_picked=0, add_hunters=0):
        #print(f"Calculating reward for {self.row}{self.col} with {self.hunters} hunters value: " + str(base_reward) + " with " + str((base_reward * self.multiplier)/(self.hunters+1)) + " ")
        
        return (base_reward * self.multiplier) / (self.hunters + percentage_of_pop_picked + add_hunters)

    def __str__(self):
        return f"({self.row}, {self.col}) {self.multiplier}x{self.hunters}"
    
initial_tiles = [
    [
        TileAttributes("G", 26, 24, 2),
        TileAttributes("G", 27, 70, 4),
        TileAttributes("G", 28, 41, 3),
        TileAttributes("G", 29, 21, 2),
        TileAttributes("G", 30, 60, 4),
    ],
    [
        TileAttributes("H", 26, 47, 3),
        TileAttributes("H", 27, 82, 5),
        TileAttributes("H", 28, 87, 5),
        TileAttributes("H", 29, 80, 5),
        TileAttributes("H", 30, 35, 3),
    ],
    [
        TileAttributes("I", 26, 73, 4),
        TileAttributes("I", 27, 89, 5),
        TileAttributes("I", 28, 100, 8),
        TileAttributes("I", 29, 90, 7),
        TileAttributes("I", 30, 17, 2),
    ],
    [
        TileAttributes("J", 26, 77, 5),
        TileAttributes("J", 27, 83, 5),
        TileAttributes("J", 28, 85, 5),
        TileAttributes("J", 29, 79, 5),
        TileAttributes("J", 30, 55, 4),
    ],
    [
        TileAttributes("K", 26, 12, 2),
        TileAttributes("K", 27, 27, 3),
        TileAttributes("K", 28, 52, 4),
        TileAttributes("K", 29, 15, 2),
        TileAttributes("K", 30, 30, 3),
    ]
]

min_tile_reward = 7500
cost_of_each_exploration = [0, 25000, 75000]


# Compute the potential profit for each tile and each possible number of expeditions
all_profits = []
for i, row in enumerate(initial_tiles):
    for tile in row:
        # Calculate initial reward assuming the given number of hunters 
        # TODO: we should consider the percentage of the population that has picked the tile too
        initial_reward = tile.real_reward(min_tile_reward, add_hunters=0)
        
        # Calculate profits subtracting the costs for each potential expedition
        # profits = [initial_reward - cost for cost in cost_of_each_exploration]
        
        # Append results including tile details
        #print(f"Tile {tile.row}{tile.col} with {tile.hunters} hunters: {initial_reward}")
        all_profits.append((f"Tile {tile.row}{tile.col}", initial_reward))

# Sort the profits by the highest profit first
all_profits.sort(key=lambda x: x[1], reverse=True)

for p in all_profits:
    print(p[0], p[1])

