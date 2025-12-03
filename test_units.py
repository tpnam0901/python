import logging

logging.root.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.root.level, format="%(name)s - %(levelname)s - %(message)s")

from configs.test_units import *
from data.test_units import *
from engine.test_units import *
from networks.test_units import *
from utils.test_units import *

if __name__ == "__main__":
    # Comment out unittest.main() when running actual training
    unittest.main()
