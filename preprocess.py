import json

target_folder = "datasets/island-tutorial"
round_num = 0

dayNum = -2

# Load the data
with open(f"{target_folder}/full.log") as f:
    data = f.readlines()
    print("Loaded data")

    isInActivitiesLog = False
    isInTradeLog = False
    header = ""
    rowNum = 0
    dayData = {}
    tradeData ={}
    zeroBasedDayNum = 0
    for i, line in enumerate(data):
        if line == "Activities log:\n":
            # ok we found the start of the activities log
            print("Found activities log")
            isInActivitiesLog = True
            rowNum = 0
            continue

        if line == "Trade History:\n":
            isInTradeLog = True
            isInActivitiesLog = False
            rowNum = 0
            print("Found trade log")
            zeroBasedDayNum = 0
            continue

        if isInActivitiesLog:
            if rowNum == 0:
                header = line
                rowNum += 1
                print("Found activities header: ", header)
                continue

            if line == "\n":
                isInActivitiesLog = False
                print("Done reading activities log row: ", i + 1)
                continue

            
            row = line.split(";")
            dayNum = int(row[0])     

            if dayNum not in dayData:
                dayData[dayNum] = []

            dayData[dayNum].append(row)
            rowNum += 1

        if isInTradeLog:
            # Read the rest of the file and parse the data since it's in format like
            """
            [
                {
                    "timestamp": 0,
                    "buyer": "",
                    "seller": "SUBMISSION",
                    "symbol": "AMETHYSTS",
                    "currency": "SEASHELLS",
                    "price": 10002,
                    "quantity": 1
                },
                {
                    "timestamp": 0,
                    "buyer": "",
                    "seller": "",
                    "symbol": "AMETHYSTS",
                    "currency": "SEASHELLS",
                    "price": 10004,
                    "quantity": 1
                },
            ]
            """

            restOfFile = data[i:]
            print("Found trade data")
        
            fullData = json.loads("".join(restOfFile)) 

            # Each day is timestamp [0, 999900] split up by that into different files
            for trade in fullData:
                if dayNum not in tradeData:
                    tradeData[dayNum] = []

                tradeData[dayNum].append(trade)


            break

    # Ok write back day data in format
    # prices_round_1_day_-2.csv

    for dayNum, rows in dayData.items():
        with open(f"{target_folder}/prices_round_{round_num}_day_{dayNum}.csv", "w") as f:
            f.write(header)
            for row in rows:
                f.write(";".join(row))


    # Then for trades write back as
    # trades_round_1_day_-2_nn.csv
    for dayNum, tradeData in tradeData.items():
        with open(f"{target_folder}/trades_round_{round_num}_day_{dayNum}_nn.csv", "w") as f:
            headers = ["timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity"]
            
            f.write(";".join(headers) + "\n")

            for trade in tradeData:
                f.write(";".join([str(trade[header]) for header in headers]) + "\n")



        

        