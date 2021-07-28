import unittest
import sapp_gym as SAPP_Env
import numpy as np


# Agent 1
num_agents1 = 1
world1 = [[ 1,  0,  0, -1,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]
goals1 = [[ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]

# Agent 1
num_agents2 = 1
world2 = [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
          [ 0,  0, -1,  1, -1,  0,  0,  0,  0,  0],
          [ 0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]
goals2 = [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]

# action: {0:NOP, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_south, 4:MOVE_WEST}
# SAPP_Env.ACTION_COST, SAPP_Env.IDLE_COST, SAPP_Env.GOAL_REWARD, SAPP_Env.COLLISION_REWARD
FULL_HELP = False

class MAPFTests(unittest.TestCase):
    # Bruteforce tests
    def test_validActions1(self):
        # SAPP_Env.SAPPEnv(self,  world0=None, goals0=None, DIAGONAL_MOVEMENT=False, SIZE=10, PROB=.2, FULL_HELP=False)
        gameEnv1 = SAPP_Env.SAPPEnv(world0=np.array(world1), goals0=np.array(goals1), DIAGONAL_MOVEMENT=False)
        validActions1 = gameEnv1._listNextValidActions()
        self.assertEqual(validActions1, [1,2])
        # With diagonal actions
        gameEnv1 = SAPP_Env.SAPPEnv(world0=np.array(world1), goals0=np.array(goals1), DIAGONAL_MOVEMENT=True)
        validActions1 = gameEnv1._listNextValidActions()
        self.assertEqual(validActions1, [1,2,5])
        
    def test_validActions2(self):
        gameEnv2 = SAPP_Env.SAPPEnv(num_agents2, world0=np.array(world2), goals0=np.array(goals2), DIAGONAL_MOVEMENT=False)
        validActions2 = gameEnv2._listNextValidActions()
        self.assertEqual(validActions2, [0])
        # With diagonal actions
        gameEnv2 = SAPP_Env.SAPPEnv(num_agents2, world0=np.array(world2), goals0=np.array(goals2), DIAGONAL_MOVEMENT=True)
        validActions2 = gameEnv2._listNextValidActions()
        self.assertEqual(validActions2, [5,6,7,8])

    
    def testIdle1(self):
        gameEnv1 = SAPP_Env.SAPPEnv(world0=np.array(world1), goals0=np.array(goals1))
        s0 = gameEnv1.world.state.copy()
        # return state, reward, done, nextActions, on_goal, blocking, valid_action
        s1, r, d, _ = gameEnv1.step(0)
        s2 = gameEnv1.world.state.copy()
        self.assertEqual(r, SAPP_Env.IDLE_COST)
        self.assertFalse(d)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def testIdle2(self):
        gameEnv2 = SAPP_Env.SAPPEnv(num_agents2, world0=np.array(world2), goals0=np.array(goals2))
        s0 = gameEnv2.world.state.copy()
        s1, r, d, _ = gameEnv2.step(0)
        s2 = gameEnv2.world.state.copy()
        self.assertEqual(r, SAPP_Env.GOAL_REWARD)
        self.assertTrue(d)
        self.assertEqual(np.sum(s0), np.sum(s2))


    def test_move_east1(self):
        gameEnv1 = SAPP_Env.SAPPEnv(world0=np.array(world1), goals0=np.array(goals1))
        s0 = gameEnv1.world.state.copy()
        # return state, reward, done, nextActions, on_goal
        s1, r, d, _ = gameEnv1.step(1)
        s2 = gameEnv1.world.state.copy()
        self.assertEqual(r, SAPP_Env.GOAL_REWARD)
        self.assertTrue(d)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_east2(self):
        gameEnv2 = SAPP_Env.SAPPEnv(num_agents2, world0=np.array(world2), goals0=np.array(goals2))
        s0 = gameEnv2.world.state.copy()
        s1, r, d, _ = gameEnv2.step(1)
        s2 = gameEnv2.world.state.copy()
        self.assertEqual(r, SAPP_Env.COLLISION_REWARD)
        self.assertTrue(d)
        self.assertEqual(np.sum(s0), np.sum(s2))


    def test_move_north1(self):
        gameEnv1 = SAPP_Env.SAPPEnv(world0=np.array(world1), goals0=np.array(goals1))
        s0 = gameEnv1.world.state.copy()
        # return state, reward, done, nextActions, on_goal
        s1, r, d, _ = gameEnv1.step(2)
        s2 = gameEnv1.world.state.copy()
        self.assertEqual(r, SAPP_Env.ACTION_COST)
        self.assertFalse(d)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_north2(self):
        gameEnv2 = SAPP_Env.SAPPEnv(num_agents2, world0=np.array(world2), goals0=np.array(goals2))
        s0 = gameEnv2.world.state.copy()
        s1, r, d, _ = gameEnv2.step(2)
        s2 = gameEnv2.world.state.copy()
        self.assertEqual(r, SAPP_Env.COLLISION_REWARD)
        self.assertTrue(d)
        self.assertEqual(np.sum(s0), np.sum(s2))


    def test_move_west1(self):
        gameEnv1 = SAPP_Env.SAPPEnv(world0=np.array(world1), goals0=np.array(goals1))
        s0 = gameEnv1.world.state.copy()
        # return state, reward, done, nextActions, on_goal
        s1, r, d, _ = gameEnv1.step(3)
        s2 = gameEnv1.world.state.copy()
        self.assertEqual(r, SAPP_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_west2(self):
        gameEnv2 = SAPP_Env.SAPPEnv(num_agents2, world0=np.array(world2), goals0=np.array(goals2))
        s0 = gameEnv2.world.state.copy()
        s1, r, d, _ = gameEnv2.step(3)
        s2 = gameEnv2.world.state.copy()
        self.assertEqual(r, SAPP_Env.COLLISION_REWARD)
        self.assertTrue(d)
        self.assertEqual(np.sum(s0), np.sum(s2))


    def test_move_south1(self):
        gameEnv1 = SAPP_Env.SAPPEnv(world0=np.array(world1), goals0=np.array(goals1))
        s0 = gameEnv1.world.state.copy()
        # return state, reward, done, nextActions, on_goal
        s1, r, d, _ = gameEnv1.step(4)
        s2 = gameEnv1.world.state.copy()
        self.assertEqual(r, SAPP_Env.COLLISION_REWARD)
        self.assertFalse(d)
        self.assertEqual(np.sum(s0), np.sum(s2))

    def test_move_south2(self):
        gameEnv2 = SAPP_Env.SAPPEnv(num_agents2, world0=np.array(world2), goals0=np.array(goals2))
        s0 = gameEnv2.world.state.copy()
        s1, r, d, _ = gameEnv2.step(4)
        s2 = gameEnv2.world.state.copy()
        self.assertEqual(r, SAPP_Env.COLLISION_REWARD)
        self.assertTrue(d)
        self.assertEqual(np.sum(s0), np.sum(s2))



if __name__ == '__main__':
    unittest.main()
