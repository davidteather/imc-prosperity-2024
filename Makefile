default:
	@echo "Please specify a target to build"

combine:
	../PyBreeder/breeder.py trader.py datamodel.py strategies/ products/ > submit_trader.py