# IMC Prosperity 2024

Writeup [here](https://dteather.com/blogs/imc-prosperity-2/)

Scaffolding from: https://github.com/MichalOkon/imc_prosperity

The code is fairly disorganized but it is what it is. We also did not have time to get working algo trades for each round which definitely costed us in the rankings since it was just me doing algotrading.

This was my first time doing a trading competition and I learned a lot. I think I could have done better if I had more time to work on it but I'm generally happy with the results :D

## Prerequisites

To install the necessary packages, use pip with the following command:

```
pip install -r requirements.txt
```

## Running the Simulation

To execute the simulation, enter the following command in your console:

```
 python -m simulator.simulator_test main <csv_filename_with_round_data> <csv_filename_with_trades_data>
```

Replace  `<csv_filename_with_round_data>`, and `<csv_filename_with_trades_data>` with the appropriate values.

For example, to run the simulation with a sample trader, you can use the following command:

```
python -m simulator.simulator_test main datasets_2023/island-data-bottle-round-4/prices_round_4_day_1.csv datasets_2023/island-data-bottle-round-4/trades_round_4_day_1_nn.csv 
```