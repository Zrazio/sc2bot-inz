import sc2
from sc2 import run_game, maps ,Race, Difficulty, position
from sc2.unit import Unit
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, CYBERNETICSCORE, STALKER, ZEALOT, FORGE, PHOTONCANNON, WARPGATERESEARCH, \
PROTOSSGROUNDWEAPONSLEVEL1,PROTOSSGROUNDWEAPONSLEVEL2,PROTOSSGROUNDWEAPONSLEVEL3,PROTOSSGROUNDARMORSLEVEL1,PROTOSSGROUNDARMORSLEVEL2,PROTOSSGROUNDARMORSLEVEL3,\
MORPH_WARPGATE, WARPGATE, WARPGATETRAIN_ZEALOT, WARPGATETRAIN_STALKER
from sc2.position import Pointlike
import cv2
import numpy
import random
import gym
from collections import deque
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy , BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory
from rl.core import Env, Processor
from keras.optimizers import Adam
from keras.callbacks import TensorBoard ,CSVLogger
import time
import threading
import math
#METAPARAMETERS

ZEALOT_VALUE = 15
MINERAL_RATE_VALUE = 20
STALKER_VALUE = 25

from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Activation, Conv2D, Dropout, BatchNormalization, ReLU, Flatten
from keras.activations import elu, relu

from keras.activations import relu, softmax
### MODEL HYPERPARAMETERS
state_size = [96,88,3]      # Our input is a stack of 3 frames
action_size = 11


class DQNetwork:
    def __init__(self):
        self.action_shape = action_size
        self.state_shape = state_size
        self.model = Sequential()
        self.model.add(Conv2D(16, (8, 8), strides=(4, 4), padding="VALID", input_shape= self.state_shape))
        self.model.add(ReLU())
        self.model.add(Conv2D(32, (4,  4), padding="VALID", strides=(2,2)))
        self.model.add(ReLU())
        self.model.add(Conv2D(32, (4,  4), padding="VALID", strides=(2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(256,activation=relu))
        self.model.add(Dense(self.action_shape, activation=relu))

    def get_model(self):
        return self.model

    def get_shapes(self):
        for layer in self.model.layers:
            print(layer.output_shape)


class PracticeBot(sc2.BotAI,Env):

    def __init__(self):
        super(PracticeBot, self).__init__()
        self.macro_action_space = [('nothing', self.do_nothing), ('build_workers',self.build_workers),
                                   ('build_pylon', self.build_pylon), ('build_ass',self.build_assimilators),
                                   ('expand',self.expand), ('army_building',self.army_buildings),
                                   ('army',self.build_army),('move_army', self.move_army_to_defend),
                                   ('attack', self.attack),('defensive_buildings',self.defensive_buildings),
                                   ('upgrade_tech', self.upgrade_tech)]
        self.running = False
        self.max_minerals = 2000
        self.max_vespene = 1000
        self.max_owned_minerals = 0
        self.action_queue = deque(maxlen=1)
        self.frames = deque([numpy.zeros((96,88)),numpy.zeros((96,88)),numpy.zeros((96,88))],maxlen=3)
        self.added_reward = 0
        self.prev_action_reward = 0
        self.warp_started = False
        self.warp_researched = False
        self.validation_imgs = 0
        self.val_step = 5
        self.act_step = 0


    def step(self, action):
        """this needs to return stacked frames, reward, done, info"""
        timeout = 0
        time.sleep(0.1)
        while not self.acted and timeout != 50:
            timeout += 1
            time.sleep(0.01)
        while len(self.frames) != 3:
            time.sleep(0.01)
        ob = numpy.stack(self.frames, axis=2)
        for _ in range(3):
            self.frames.pop()
        self.action_queue.append(action)
        #print(action)
        self.acted = False
        rew = self.get_prev_reward()
        self.added_reward = 0

        return ob, rew, not self.running, {}

    def on_end(self, game_result):
        self.acted = True
        self.running = False
        self.acted = True

    def on_start(self):
        self.warp_researched = False
        self.warp_started = False
        self.running = True
        self.acted = True
        self.added_reward = 0

    def get_prev_reward(self):
        minerals_reward = self.state.score.collection_rate_minerals * 2 * 0.1
        minerals_reward += 50 if self.minerals < self.max_owned_minerals/3 else -50

        army_reward = (-10 * self.time)* 0.1 if len(self.units(STALKER) + self.units(ZEALOT)) == 0 \
            else len(self.units(ZEALOT)) * ZEALOT_VALUE + len(self.units(STALKER)) * STALKER_VALUE

        supply_reward = - 100 if self.supply_left < int(self.supply_cap * 0.1) else self.supply_used * 10
        supply_reward -= 2 * (self.supply_cap - self.supply_used -20)

        action_reward = self.state.score.killed_value_units + self.state.score.killed_value_structures
        if action_reward > 0 :
            action_reward = 0

        probe_reward = - sum([i.ideal_harvesters - i.assigned_harvesters for i in self.townhalls]) * 20

        cur_action_reward = action_reward - self.prev_action_reward
        self.prev_action_reward = action_reward

        #queued_reward = self.added_reward + 1

        print("Rewards : minerals_reward : {}, army_reward = {} , supply_reward = {}, probe_reward = {}, prev_action_reward = {} , generic_reward = {}".format(
            minerals_reward, army_reward, supply_reward, probe_reward, self.prev_action_reward, self.added_reward
        ))

        return self.added_reward + minerals_reward + army_reward + supply_reward + cur_action_reward + probe_reward

    async def on_unit_destroyed(self, unit_tag):
        if unit_tag in [PROBE]:
            self.added_reward -= 30
        if unit_tag in [ZEALOT, STALKER]:
            self.added_reward -=35

    async def on_unit_created(self, unit: Unit):
        if unit.type_id in [ZEALOT,STALKER]:
            furthest_building = self.units().structure.closest_to(self.enemy_start_locations[0]).position.towards(self.enemy_start_locations[0],5)
            await self.do(unit.attack(furthest_building))

    def reset(self):

        while not self.running or len(self.frames) != 3:
            time.sleep(0.1)

        ob = numpy.stack(self.frames,axis=2)
        #print(self.running)
        #print(ob.shape)
        return ob

    def render(self, mode='human', close=False):
        pass

    @property
    def get_macro_action_space_size(self):
        return self.macro_action_space

    def enemy_units_close_to_structures(self):
        retval = None
        for i in self.units():
            units = self.known_enemy_units.closer_than(20,i)
            if len(units):
                retval = retval + units if retval is not None else units
        return retval if retval is not None else []

    async def on_step(self, iteration):
        await self.distribute_workers()
        if self.minerals > self.max_owned_minerals:
            self.max_owned_minerals = self.minerals
        while(len(self.action_queue) > 0):
            action_no = self.action_queue.pop()
            #print("Action number: {}".format(action_no))
            action = self.macro_action_space[action_no]

            print("Performing action: {}".format(action[0]))
            await action[1]()
            self.acted = True
        if self.warp_started:
            if WARPGATERESEARCH in self.state.upgrades:
                self.warp_researched= True
                self.warp_started = False
        await self.draw_map()

    async def move_army_to_defend(self):
        print ("Moving army")
        if len(self.units(NEXUS)) > 1:
            furthest_building = self.units().structure.closest_to(self.enemy_start_locations[0]).position.towards(self.enemy_start_locations[0],5)
            attack_units = (self.units(ZEALOT).idle.further_than(5, furthest_building) + self.units(STALKER).idle.further_than(5, furthest_building))
            for unit in attack_units:
                self.added_reward += 10
                await self.do(unit.attack(furthest_building))

        else:
            ramp_position = self.main_base_ramp.top_center
            attack_units = (self.units(ZEALOT).idle.further_than(5, ramp_position) + self.units(STALKER).idle.further_than(5, ramp_position))
            for unit in attack_units:
                self.added_reward += 10
                await self.do(unit.attack(ramp_position))

    async def do_nothing(self):
        print("Doing nothing")

    async def build_workers(self):
        print("Building probes")
        for nexus in self.units(NEXUS).ready:
            needed_workers = nexus.ideal_harvesters + sum([i.ideal_harvesters for i in
                                                            self.units(ASSIMILATOR).closer_than(15.0,nexus)])
            assigned_workers = nexus.assigned_harvesters + sum([i.assigned_harvesters for i in
                                                            self.units(ASSIMILATOR).closer_than(15.0,nexus)])
            if self.can_afford(PROBE) and assigned_workers < needed_workers:
                self.added_reward += 50
                await self.do(nexus.train(PROBE))
            else:
                self.added_reward -= 50

    async def build_pylon(self):
        print("Building pylon")
        if self.can_afford(PYLON):
            structures = self.units().structure
            if structures is not None:
                if self.can_afford(PYLON):
                    if self.supply_cap == 200:
                        self.added_reward -= 20
                    else:
                        self.added_reward += 10
                    await self.build(PYLON, near=random.choice(structures), max_distance=100, placement_step=3)
                else:
                    self.added_reward -= 50

    async def build_assimilators(self):
        print("Building assimilators")
        for nexus in self.units(NEXUS).ready:
            geysers = self.state.vespene_geyser.closer_than(15.0,nexus)
            for geyser in [j for j in geysers if not self.units(ASSIMILATOR).closer_than(1,j).exists]:
                if not self.can_afford(ASSIMILATOR):
                    break
                builder = self.select_build_worker(geyser)
                if builder is not None:
                    self.added_reward += 100
                    await self.do(builder.build(ASSIMILATOR,geyser))

    async def expand(self):
        print("Expanding")
        if self.units(NEXUS).amount < 2 and self.can_afford(NEXUS):
                self.added_reward += 200
                await self.expand_now()
        else:
            self.added_reward -= 50

    async def army_buildings(self):
        print("Building army buildings")
        if self.warp_researched:
            if self.units(GATEWAY).ready.exists:
                for i in self.units(GATEWAY):
                    await self.do(i(MORPH_WARPGATE))

        if self.units(PYLON).ready.exists:
            pylon = random.choice(self.units(PYLON))
            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE).exists:
                if not self.already_pending(CYBERNETICSCORE) and self.can_afford(CYBERNETICSCORE):
                    self.added_reward += 100
                    await self.build(CYBERNETICSCORE,near=pylon, placement_step=5)
            elif self.units(GATEWAY).amount < 4 and self.can_afford(GATEWAY):
                pylon = self.units(PYLON).closest_to(self.townhalls.first)
                self.added_reward += 50
                await self.build(GATEWAY,near=pylon,placement_step=3)
        else:
            self.added_reward -= 100

    async def defensive_buildings(self):
        print("Building defensive buildings")
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).closest_to(self.units().closest_to(self.enemy_start_locations[0]))
            if self.units(FORGE).ready.exists:
                if self.can_afford(PHOTONCANNON):
                    await self.build(PHOTONCANNON, near=pylon , placement_step=3)
                else:
                    self.added_reward -= 50
            else:
                if self.can_afford(FORGE):
                    await self.build(FORGE, near=self.townhalls.first,placement_step=3)
                else:
                    self.added_reward -= 50

    async def upgrade_tech(self):
        if not self.warp_researched:
            if self.units(CYBERNETICSCORE).ready.noqueue.exists:
                core = self.units(CYBERNETICSCORE).noqueue.random
                if self.can_afford(WARPGATERESEARCH):
                    self.warp_started = True

                    await self.do(core.research(WARPGATERESEARCH))

        if self.units(FORGE).ready.noqueue.exists:
            forge = self.units(FORGE).noqueue.random
            if PROTOSSGROUNDARMORSLEVEL1 not in self.state.upgrades:
                if self.can_afford(PROTOSSGROUNDARMORSLEVEL1):
                    await self.do(forge.research(PROTOSSGROUNDARMORSLEVEL1))
            elif PROTOSSGROUNDWEAPONSLEVEL1 not in self.state.upgrades:
                if self.can_afford(PROTOSSGROUNDWEAPONSLEVEL1):
                    await self.do(forge.research(PROTOSSGROUNDWEAPONSLEVEL1))
            elif PROTOSSGROUNDARMORSLEVEL2 not in self.state.upgrades:
                if self.can_afford(PROTOSSGROUNDARMORSLEVEL2):
                    await self.do(forge.research(PROTOSSGROUNDARMORSLEVEL2))
            elif PROTOSSGROUNDWEAPONSLEVEL2 not in self.state.upgrades:
                if self.can_afford(PROTOSSGROUNDWEAPONSLEVEL2):
                    await self.do(forge.research(PROTOSSGROUNDWEAPONSLEVEL2))
            elif PROTOSSGROUNDARMORSLEVEL3 not in self.state.upgrades:
                if self.can_afford(PROTOSSGROUNDARMORSLEVEL3):
                    await self.do(forge.research(PROTOSSGROUNDARMORSLEVEL3))
            elif PROTOSSGROUNDWEAPONSLEVEL3 not in self.state.upgrades:
                if self.can_afford(PROTOSSGROUNDWEAPONSLEVEL3):
                    await self.do(forge.research(PROTOSSGROUNDWEAPONSLEVEL3))

    async def build_army(self):
        print("Building army")
        if self.units(WARPGATE).exists:
            if len(self.units(NEXUS)) > 1:
                position = self.units(PYLON).closest_to(self.enemy_start_locations[0]).position.towards(
                    self.enemy_start_locations[0], 5)
            else:
                position = self.units(PYLON).closest_to(self.main_base_ramp.top_center).position


            for warpgate in self.units(WARPGATE).ready:
                abilities = await self.get_available_abilities(warpgate)
                # all the units have the same cooldown anyway so let's just look at ZEALOT
                if WARPGATETRAIN_ZEALOT in abilities:
                    pos = position.to2.random_on_distance(4)
                    placement = await self.find_placement(WARPGATETRAIN_STALKER, pos, placement_step=1)
                    if placement is None:
                        # return ActionResult.CantFindPlacementLocation
                        print("can't place")
                        return
                    await self.do(warpgate.warp_in(STALKER, placement))

        if self.units(GATEWAY).exists:
            for gateway in self.units(GATEWAY).ready.noqueue:
                if self.units(CYBERNETICSCORE).exists and not self.already_pending(CYBERNETICSCORE):
                    if self.units(STALKER).amount < 2 * self.units(ZEALOT).amount:
                        if self.can_afford(STALKER):
                            self.added_reward += 100
                            await self.do(gateway.train(STALKER))

                    else:
                        if self.can_afford(ZEALOT):
                            self.added_reward += 50
                            await self.do(gateway.train(ZEALOT))
                elif self.can_afford(ZEALOT):
                    self.added_reward += 50
                    await self.do(gateway.train(ZEALOT))
        else:
            self.added_reward -= 50

    async def draw_map(self):
        raw_map = numpy.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3))
        learn_map = numpy.zeros((self.game_info.map_size[1], self.game_info.map_size[0]))

        unit_representations = {
            PROBE: (1,(0,255,0), 1),
            NEXUS: (2, (255, 0, 0), 2),
            ZEALOT: (1, (0, 230, 0), 3),
            STALKER: (1, (50, 200, 0), 4),
            ASSIMILATOR: (2, (0, 180, 0), 5),
            CYBERNETICSCORE: (2, (0, 150, 0), 6),
            GATEWAY: (2, (150, 150, 0), 7),
            PYLON: (1, (100, 100, 0), 8),
            FORGE: (2, (120,120,0),12),
            PHOTONCANNON: (2, (140,140,0),13)
        }

        for type in sorted(unit_representations.keys(), key= lambda x : unit_representations[x][0], reverse=True):
            for unit in self.units(type):
                cv2.circle(raw_map, (int(unit.position[0]),int(unit.position[1])), unit_representations[type][0], unit_representations[type][1], -1)
                cv2.circle(learn_map, (int(unit.position[0]),int(unit.position[1])), unit_representations[type][0], unit_representations[type][2], -1)
        cv2.circle
        for i in self.known_enemy_units:
            cv2.circle(learn_map, (int(i.position[0]),int(i.position[1])), 1, 100, -1)
            cv2.circle(raw_map , (int(i.position[0]),int(i.position[1])), 1, (0,255,255), -1)


        for i in self.known_enemy_structures:
            cv2.circle(learn_map, (int(i.position[0]),int(i.position[1])), 2, 101, -1)
            cv2.circle(raw_map, (int(i.position[0]),int(i.position[1])), 2, (0,255,255), -1)


        cv2.line(raw_map,(0,0),(0,int(self.game_info.map_size[0] * (self.minerals/self.max_minerals))),(255,255,0))
        cv2.line(learn_map,(0,0),(0,int(self.game_info.map_size[0] * (self.minerals/self.max_minerals))), 9)


        cv2.line(raw_map,(1,0),(1,int(self.game_info.map_size[0] * (self.vespene/self.max_vespene))),(0,255,0))
        cv2.line(learn_map, (1, 0), (1, int(self.game_info.map_size[0] * (self.vespene / self.max_vespene))), 10)

        cv2.line(raw_map,(2,0),(1,int(self.game_info.map_size[0] * (self.time/8000))),(0,0,255))
        cv2.line(learn_map, (2, 0), (1, int(self.game_info.map_size[0] * (self.time/8000))), 11)

        flp = cv2.flip(raw_map, 0)
        map = cv2.resize(flp, None, fx=2, fy=2)
        self.frames.append(learn_map)

        #if self.val_step == self.act_step:
        #    if self.validation_imgs < 15:
        #        cv2.imwrite('val_imgs/val_im_{}.i'.format(self.validation_imgs),learn_map)
        #        self.validation_imgs += 1
        #    self.act_step = 0
        #else:
        #    self.act_step += 1


            #print(learn_map)
        #print(map)
        cv2.imshow('Map', map)
        cv2.waitKey(1)
        #cv2.imshow('learn', learn_map)

    async def attack(self):
        attack_units = self.units(ZEALOT).idle + self.units(STALKER).idle
        if len(attack_units) > 10:
            if len(attack_units) == 0:
                self.added_reward -= 50
            elif len(self.known_enemy_units) > 0:
                for unit in attack_units:
                    await self.do(unit.attack(self.known_enemy_units.closest_to(self.units(NEXUS).random.position).position))
            else:
                for unit in attack_units:
                    await self.do(unit.attack(self.enemy_start_locations[0]))

    async def scout(self):
        self.state.visibility.save_image('visibility.jpg')

        if not len(self.units(STALKER) + self.units(ZEALOT)):
            probe = random.choice(self.units(PROBE).ready)
            await self.do(probe.move(self.enemy_start_locations[0]))



def start_game(bot):
    print("Starting game")
    return run_game(maps.get("Simple64"),[
        Bot(Race.Protoss, bot),
        Computer(Race.Terran, Difficulty.Normal)
        ],realtime=False)

def run_learning(bot):
    #agent = DQNAgent(DQNetwork().get_model(), policy=EpsGreedyQPolicy())
    memory = SequentialMemory(limit=200000, window_length=1)
    policy = EpsGreedyQPolicy(eps=2)
    dqn = DQNAgent(model=DQNetwork().get_model(), nb_actions=11, memory=memory, processor=MyProcessor(), policy=policy, batch_size=1)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.load_weights('dqn_{}_weights.h5f'.format('sc2_inz_11_new_net'))
    log_nr = 1
    tb_log_dir = 'logs/tmp{}'.format(log_nr)
    tb_callback = TensorBoard(histogram_freq=0)
    log_callback = CSVLogger('learning_log{}.csv'.format(log_nr))

    for _ in range(10):
        tb_callback.log_dir = tb_log_dir
        dqn.fit(bot, nb_steps=10000, visualize=True, verbose=2, callbacks=[tb_callback, log_callback],log_interval=100)
        #policy.eps -= policy.eps * 0.5
        #dqn.policy = policy
        #print("#######################################NEWPOLICY: {}".format(policy.eps))
        dqn.save_weights('dqn_{}_weights.h5f'.format('sc2_inz_11_new_net'), overwrite=True)
        log_nr +=1
        tb_log_dir = 'logs/tmp{}'.format(log_nr)

    dqn.save_weights('dqn_{}_weights.h5f'.format('sc2_inz_11_new_net'), overwrite=True)

    dqn.test(bot, nb_episodes=10, visualize=True)

class MyProcessor(Processor):
    def process_state_batch(self, batch):
        return batch[0]


bot = PracticeBot()
thread = threading.Thread(target=run_learning, args=(bot,))
thread.start()
f = open('results.txt', 'w')
for i in range(1500):
    try:
        bot.running = False
        time.sleep(3)
        res = start_game(bot)
        print(res)
        f.write('Game {} , result : {} \n'.format(i, 'win' if res else 'lost'))
        f.close()
        f = open('results.txt', 'aw')
        bot.running = False
    except:
        bot.running = False
