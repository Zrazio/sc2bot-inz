import sc2
from sc2 import run_game, maps ,Race, Difficulty, position
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, CYBERNETICSCORE, STALKER, ZEALOT
import cv2
import numpy
import random
import rl

class PracticeBot(sc2.BotAI):
    max_minerals = 2000
    max_vespene = 1000
    macro_action_space = []
    micro_action_space = []
    score = 0

    def __init__(self):
        self.macro_action_space = [self.do_nothing,self.build_pylon,self.build_assimilators,
                                   self.expand, self.army_buildings, self.build_army]

    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylon()
        await self.build_assimilators()
        await self.expand()
        await self.army_buildings()
        await self.build_army()
        await self.draw_map()
        print(self.state.score.score)

    async def macro_decision(self):
        choice = random.randint(0,7)
        self.macro_action_space[choice]()

    @property
    def reward(self):
        return self.state.score.score

    async def build_workers(self):
        for nexus in self.units(NEXUS).ready.noqueue:
            needed_workers = nexus.ideal_harvesters + sum([i.ideal_harvesters for i in
                                                            self.units(ASSIMILATOR).closer_than(15.0,nexus)])
            assigned_workers = nexus.assigned_harvesters + sum([i.assigned_harvesters for i in
                                                            self.units(ASSIMILATOR).closer_than(15.0,nexus)])
            if self.can_afford(PROBE) and assigned_workers < needed_workers:
                await self.do(nexus.train(PROBE))

    async def build_pylon(self):
        if self.supply_left < 4 and self.can_afford(PYLON) and not self.already_pending(PYLON):
            structures = self.units().structure
            if structures is not None:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=random.choice(structures),max_distance=100)

    async def build_assimilators(self):
        for nexus in self.units(NEXUS).ready:
            geysers = self.state.vespene_geyser.closer_than(15.0,nexus)
            for geyser in [j for j in geysers if not self.units(ASSIMILATOR).closer_than(1,j).exists]:
                if not self.can_afford(ASSIMILATOR):
                    break
                builder = self.select_build_worker(geyser)
                if builder is not None:
                    await self.do(builder.build(ASSIMILATOR,geyser))

    async def expand(self):
        if self.units(NEXUS).amount < 3 and self.can_afford(NEXUS):
                await self.expand_now()

    async def army_buildings(self):
        if self.units(PYLON).ready.exists:
            pylon = random.choice(self.units(PYLON))
            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE).exists:
                if not self.already_pending(CYBERNETICSCORE) and self.can_afford(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE,near=pylon)
            elif self.units(GATEWAY).amount < 4 and self.can_afford(GATEWAY):
                pylon = self.units(PYLON).closest_to(self.townhalls.first)
                await self.build(GATEWAY,near=pylon)

    async def build_army(self):
        if self.units(GATEWAY).exists:
            for gateway in self.units(GATEWAY).ready.noqueue:
                if self.units(CYBERNETICSCORE).exists and not self.already_pending(CYBERNETICSCORE):
                    if self.units(STALKER).amount < 2 * self.units(ZEALOT).amount:
                        if self.can_afford(STALKER):
                            await self.do(gateway.train(STALKER))
                    else:
                        if self.can_afford(ZEALOT):
                            await self.do(gateway.train(ZEALOT))
                elif self.can_afford(ZEALOT):
                    await self.do(gateway.train(ZEALOT))

    async def draw_map(self):
        raw_map = numpy.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3))
        learn_map = numpy.zeros((self.game_info.map_size[1], self.game_info.map_size[0]))
        print(raw_map.shape)

        unit_representations = {
            PROBE: (1,(0,255,0), 1),
            NEXUS: (10, (255, 0, 0), 2),
            ZEALOT: (2, (0, 230, 0), 3),
            STALKER: (2, (50, 200, 0), 4),
            ASSIMILATOR: (4, (0, 180, 0), 5),
            CYBERNETICSCORE: (3, (0, 150, 0), 6),
            GATEWAY: (5, (150, 150, 0), 7),
            PYLON: (2, (100, 100, 0), 8)
        }

        for i in range(self.game_info.map_size[0]):
            for j in range(self.game_info.map_size[1]):
                if self.state.visibility[i,j]:
                    raw_map[j,i] = [255,255,255]

        for type in sorted(unit_representations.keys(), key= lambda x : unit_representations[x][0], reverse=True):
            for unit in self.units(type):
                cv2.circle(raw_map, (int(unit.position[0]),int(unit.position[1])), unit_representations[type][0], unit_representations[type][2], -1)
                cv2.circle(learn_map, (int(unit.position[0]),int(unit.position[1])), unit_representations[type][0], unit_representations[type][2], -1)


        #if self.minerals > self.max_minerals :
        #    self.max_minerals = self.minerals

        cv2.line(raw_map,(0,0),(0,int(self.game_info.map_size[0] * (self.minerals/self.max_minerals))),(255,255,0))
        cv2.line(learn_map,(0,0),(0,int(self.game_info.map_size[0] * (self.minerals/self.max_minerals))), 9)


        #if self.vespene > self.max_vespene :
        #    self.max_vespene = self.vespene

        cv2.line(raw_map,(1,0),(1,int(self.game_info.map_size[0] * (self.vespene/self.max_vespene))),(0,255,0))
        cv2.line(learn_map, (1, 0), (1, int(self.game_info.map_size[0] * (self.vespene / self.max_vespene))), 10)

        flp = cv2.flip(learn_map, 0)
        print(flp.shape)
        map = cv2.resize(flp, None, fx=2, fy=2)
        cv2.imshow('Map',map)
        cv2.waitKey(1)


run_game(maps.get("Simple64"),[
    Bot(Race.Protoss, PracticeBot()),
    Computer(Race.Terran, Difficulty.Easy)
    ],realtime=False,)