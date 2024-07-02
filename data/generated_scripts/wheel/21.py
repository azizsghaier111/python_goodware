# Required Libraries
import numpy as np
import mock
import pytorch_lightning as pl

# Ensure Mechanical advantage
class Mechanical:

    def __init__(self, eff_effort, eff_resistance):
        self.effort = eff_effort
        self.resistance = eff_resistance
        self.advantage = None

    def calculate_advantage(self):
        self.advantage = self.resistance/self.effort
        return self.advantage

# Ensure Directional control
class Directional:
    
    def __init__(self):
        self.current_direction = None
        self.direction = {"NORTH":0, "SOUTH":180, "EAST":90, "WEST": 270}

    def set_direction(self, direction):
        self.current_direction =  self.direction[direction]
        return self.current_direction

# Ensure Speed control
class Speed:

    def __init__(self,initial_speed):
        self.speed = initial_speed

    def get_speed(self):
        return self.speed
   
    def increase_speed(self, increase_value):
        self.speed += increase_value
        return self.speed

    def decrease_speed(self, decrease_value):
        self.speed -= decrease_value
        return self.speed

def main():
    # Mocking the objects
    mechanical_mock = mock.create_autospec(Mechanical)
    directional_mock = mock.create_autospec(Directional)
    speed_mock = mock.create_autospec(Speed)

    # Mechanical advantage
    mechanical_mock.effort = 20
    mechanical_mock.resistance = 100
    mechanical_advantage = mechanical_mock.calculate_advantage         
    print(f"Mechanical Advantage: {mechanical_advantage}")

    # Directional control
    directional_mock.current_direction = "NORTH"
    current_direction = directional_mock.set_direction         
    print(f"Current Direction: {current_direction}")
    
    # Speed control
    speed_mock.speed = 50
    new_speed = speed_mock.increase_speed(10)
    print(f"New Speed: {new_speed}")

if __name__ == "__main__":
    main()