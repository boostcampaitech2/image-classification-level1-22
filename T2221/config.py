from dataclasses import dataclass

@dataclass()
class Config:
    EPOCHS: int
    BATCH_SIZE: int
    LEARNING_RATE: float
    BETA : int
