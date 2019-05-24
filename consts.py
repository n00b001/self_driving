SAVE_LIMIT = 10 * 75
# IMAGE_SIZE = 224
IMAGE_SIZE = 96
IMAGE_DEPTH = 3
DATA_DIR = "data"
# DATA_DIR = "D:/old"
GAME_WIDTH = 800
GAME_HEIGHT = 600
LOG_DIR = "logs"
MODEL_DIR = "models"
SHUFFLE_BUFFER = 10_000
MAX_TRAINING_STEPS = 5_000_000
# DEFAULT_ARCHITECTURE = [64, 32, 64, 64, 32]
DEFAULT_ARCHITECTURE = [128, 64]  # 90%
# BATCH_SIZE = 7
# BATCH_SIZE = 5
BATCH_SIZE = 32
# BATCH_SIZE = 512
EPOCHS = 1
# EPOCHS = 2
FINE_TUNE_EPOCHS = 10
# FINE_TUNE_EPOCHS = 2
STEPS_PER_EPOCH = 200_000
VAL_STEPS = 100_000//BATCH_SIZE
CACHE_LOCATION = 'D:/tf_cache/'
LEARNING_RATE = 0.001
