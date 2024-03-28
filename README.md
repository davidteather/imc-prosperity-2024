# IMC Prosperity 2024

Scaffolding from: https://github.com/MichalOkon/imc_prosperity

## Prerequisites

To install the necessary packages, use pip with the following command:

```
pip install -r requirements.txt
```

## Running the Simulation

To execute the simulation, enter the following command in your console:

```
 python -m simulator.simulator_test trader <csv_filename_with_round_data> <csv_filename_with_trades_data>
```

Replace  `<csv_filename_with_round_data>`, and `<csv_filename_with_trades_data>` with the appropriate values.

For example, to run the simulation with a sample trader, you can use the following command:

```
python -m simulator.simulator_test trader datasets_2023/island-data-bottle-round-4/prices_round_4_day_1.csv datasets_2023/island-data-bottle-round-4/trades_round_4_day_1_nn.csv 
```

## Code Organization

The code is organized as follows:

- `trader.py`: Implements the trader logic
- `datamodel.py`: Defines basic data models used in trading
- `products`: This directory holds the implementation of specific products
- `strategies`: This directory contains implementations of strategies employed across multiple products
- `helpers`: This directory houses helper functions utilized during the parameter tuning process
- `simulator`: This directory comprises the code for the game simulator that allows for strategy testing without official game access

## Write Up

The writeup for our strategy is here

### Tutorial

