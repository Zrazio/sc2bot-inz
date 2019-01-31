import sc2
from sc2 import run_game, maps, Race, Difficulty, PlayerType
from sc2.player import Bot, Computer, Human 


run_game(maps.get("(2)CatalystLE"), [
    Human(Race.Protoss),
    Computer(Race.Terran, Difficulty.Easy)
], realtime=True)
