import gym
from gym import spaces
import numpy as np
from collections import OrderedDict
import sys
from matplotlib.colors import hsv_to_rgb
import random
import math
import copy
from gym.envs.classic_control import rendering        


'''
    Observation: (position maps of current agent, current goal, obstacles), vector to goal (vx, vy, norm_v)
        
    Action space: (Tuple)
        	agent_id: positive integer
        action: {0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST,
        5:NE, 6:SE, 7:SW, 8:NW}
    Reward: ACTION_COST for each action, GOAL_REWARD when robot arrives at target
'''


ACTION_COST, IDLE_COST, GOAL_REWARD, COLLISION_REWARD = -0.1, -0.2, 1.0, -1.0
opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
dirDict = {0:(0,0),1:(0,1),2:(1,0),3:(0,-1),4:(-1,0),5:(1,1),6:(1,-1),7:(-1,-1),8:(-1,1)}
actionDict={v:k for k,v in dirDict.items()}


class State(object):
    '''
    State.
    Implemented as 2 2d numpy arrays.
    first one "state":
        static obstacle: -1
        empty: 0
        agent = positive integer (agent_id)
    second one "goals":
        agent goal = positive int(agent_id)
    '''

    def __init__(self, world0, goals, diagonal):
        assert(len(world0.shape) == 2 and world0.shape==goals.shape)
        self.state                    = world0.copy()
        self.goals                    = goals.copy()
        self.agent_pos, self.agent_past, self.agent_goal = self.scanForAgent()
        self.diagonal                 = diagonal # true or false

    def scanForAgent(self):
        agent_pos  = (-1,-1)
        agent_last = (-1,-1)
        agent_goal = (-1,-1)
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if(self.state[i,j]>0):
                    agent_pos  = (i,j)
                    agent_last = (i,j)
                if(self.goals[i,j]>0):
                    agent_goal = (i,j)

        assert(agent_pos != (-1,-1) and agent_goal != (-1,-1))
        assert(agent_pos == agent_last)
        return agent_pos, agent_last, agent_goal

    def getPos(self):
        return self.agent_pos

    def getPastPos(self):
        return self.agent_past

    def getGoal(self):
        return self.agent_goal
    
    #try to move agent and return the status
    def moveAgent(self, direction):
        ax, ay = self.agent_pos

        # Not moving is always allowed
        if(direction==(0,0)):
            self.agent_past = self.agent_pos
            return 0

        # Otherwise, let's look at the validity of the move
        dx,dy = direction[0], direction[1]
        if(ax+dx >= self.state.shape[0] or ax+dx < 0 or 
           ay+dy >= self.state.shape[1] or ay+dy < 0): #out of bounds
            return -1
        if(self.state[ax+dx,ay+dy] < 0): #collide with static obstacle
            return -2

        # No collision: we can carry out the action
        self.state[ax,ay] = 0
        self.state[ax+dx,ay+dy] = 1
        self.agent_past   = self.agent_pos
        self.agent_pos    = (ax+dx,ay+dy)
        if self.goals[ax+dx,ay+dy] == 1: # reached goal
            return 1

        # none of the above
        return 0

    # try to execture action and return whether action was executed or not and why
    #returns:
    #     1: action executed and reached goal
    #     0: action executed
    #    -1: out of bounds
    #    -2: collision with wall
    def act(self, action):
        # 0     1  2  3  4 
        # still N  E  S  W
        direction = self.getDir(action)
        moved = self.moveAgent(direction)
        return moved

    def getDir(self,action):
        return dirDict[action]

    def getAction(self,direction):
        return actionDict[direction]


class SAPPEnv(gym.Env):

    # Initialize env
    def __init__(self, observation_size=11, world0=None, goals0=None, DIAGONAL_MOVEMENT=False, SIZE=(10,40), PROB=(0,.5)):
        """
        Args:
            DIAGONAL_MOVEMENT: if the agents are allowed to move diagonally
            SIZE: size of a side of the square grid
            PROB: range of probabilities that a given block is an obstacle
        """

        # Initialize member variables
        self.observation_size   = observation_size
        self.SIZE               = SIZE
        self.PROB               = PROB
        self.fresh              = True
        self.finished           = False
        self.DIAGONAL_MOVEMENT  = DIAGONAL_MOVEMENT

        # Initialize data structures
        self._setWorld(world0,goals0)
        if DIAGONAL_MOVEMENT:
            self.action_space = spaces.Tuple([spaces.Discrete(1), spaces.Discrete(9)])
        else:
            self.action_space = spaces.Tuple([spaces.Discrete(1), spaces.Discrete(5)])
        self.viewer           = None

    def isConnected(self,world0):
        sys.setrecursionlimit(10000)
        world0 = world0.copy()

        def firstFree(world0):
            for x in range(world0.shape[0]):
                for y in range(world0.shape[1]):
                    if world0[x,y] == 0:
                        return x,y

        def floodfill(world,i,j):
            sx,sy=world.shape[0],world.shape[1]
            if(i < 0 or i >= sx or j < 0 or j >= sy):#out of bounds, return
                return
            if(world[i,j] == -1):return
            world[i,j] = -1
            floodfill(world,i+1,j)
            floodfill(world,i,j+1)
            floodfill(world,i-1,j)
            floodfill(world,i,j-1)

        i,j = firstFree(world0)
        floodfill(world0,i,j)

        if np.any(world0 == 0):
            return False
        else:
            return True

    def getObstacleMap(self):
        return (self.world.state == -1).astype(int)
    
    def _setWorld(self, world0=None, goals0=None):

        def getConnectedRegion(world,x,y):
            sys.setrecursionlimit(1000000)
            '''returns a list of tuples of connected squares to the given tile
            this is memoized with a dict'''
            visited=set()
            sx,sy=world.shape[0],world.shape[1]
            work_list=[(x,y)]
            while len(work_list)>0:
                (i,j)=work_list.pop()
                if(i < 0 or i >= sx or j < 0 or j >= sy): # out of bounds, return
                    continue
                if(world[i,j] == -1):
                    continue#crashes
                if (i,j) in visited:continue
                visited.add((i,j))
                work_list.append((i+1,j))
                work_list.append((i,j+1))
                work_list.append((i-1,j))
                work_list.append((i,j-1))
            return visited

        #defines the State object, which includes initializing goals and agents
        #sets the world to world0 and goals, or if they are None randomizes world
        if not (world0 is None):
            if goals0 is None:
                raise Exception("you gave a world with no goals!")

            self.initial_world = world0
            self.initial_goals = goals0
            self.world = State(world0, goals0, self.DIAGONAL_MOVEMENT)
            # self.world.state, self.world.goals
            return

        #otherwise we have to randomize the world
        #RANDOMIZE THE STATIC OBSTACLES
        prob  = np.random.triangular(self.PROB[0],.33*self.PROB[0]+.66*self.PROB[1],self.PROB[1])
        size  = np.random.choice([self.SIZE[0], self.SIZE[0]*.5+self.SIZE[1]*.5, self.SIZE[1]], p=[.5,.25,.25])
        world = - (np.random.rand(int(size),int(size)) < prob).astype(int)

        #RANDOMIZE THE POSITION OF THE AGENT
        agent_placed = False
        while not agent_placed:
            x, y = np.random.randint(0,world.shape[0]),np.random.randint(0,world.shape[1])
            if(world[x,y] == 0):
                world[x,y]   = 1
                agent_pos    = (x,y)
                agent_placed = True
        
        #RANDOMIZE THE GOALS OF AGENT
        goals = np.zeros(world.shape).astype(int)
        goal_placed = False
        while not goal_placed:
            valid_tiles = getConnectedRegion(world, agent_pos[0], agent_pos[1])
            x, y        = random.choice(list(valid_tiles))
            if(goals[x,y] == 0 and world[x,y] != -1):
                goals[x,y]  = 1
                goal_placed = True

        self.initial_world = world
        self.initial_goals = goals
        self.world = State(world, goals, self.DIAGONAL_MOVEMENT)

    # Returns an observation of an agent
    def _observe(self):
        top_left     = (self.world.getPos()[0] - self.observation_size//2, self.world.getPos()[1] - self.observation_size//2)
        bottom_right = (top_left[0] + self.observation_size, top_left[1] + self.observation_size)
        obs_shape    = (self.observation_size, self.observation_size)

        obs_map              = np.zeros(obs_shape)
        pos_map              = np.zeros(obs_shape)
        goal_map             = np.zeros(obs_shape)

        for i in range(top_left[0],top_left[0]+self.observation_size):
            for j in range(top_left[1],top_left[1]+self.observation_size):
                if i >= self.world.state.shape[0] or i < 0 or j >= self.world.state.shape[1] or j < 0:
                    # out of bounds, just treat as an obstacle
                    obs_map[i-top_left[0],j-top_left[1]]  = 1
                    continue
                if self.world.state[i,j] == -1:
                    # obstacles
                    obs_map[i-top_left[0],j-top_left[1]]  = 1
                if self.world.state[i,j] == 1:
                    # agent's position
                    pos_map[i-top_left[0],j-top_left[1]]  = 1
                if self.world.goals[i,j] == 1:
                    # agent's goal
                    goal_map[i-top_left[0],j-top_left[1]] = 1

        # Goal vector
        dx  = self.world.getGoal()[0] - self.world.getPos()[0]
        dy  = self.world.getGoal()[1] - self.world.getPos()[1]
        mag = (dx**2 + dy**2) ** .5
        if mag != 0:
            dx = dx/mag
            dy = dy/mag

        return ([obs_map,pos_map,goal_map], [dx,dy,mag]) # (3,11,11) (1,3)

    # Resets environment
    def reset(self, world0=None, goals0=None):
        self.finished = False

        # Initialize data structures
        self._setWorld(world0,goals0)
        self.fresh = True
        
        if self.viewer is not None:
            self.viewer = None

        return self._observe(), self._listNextValidActions()

    # Executes an action and returns new state, reward, done, and next valid actions
    def step(self, action):
        self.fresh = False
        n_actions = 9 if self.DIAGONAL_MOVEMENT else 5

        # Check action input
        assert action in range(n_actions), 'Invalid action'

        # Execute action & determine reward
        action_status = self.world.act(action)
        #     1: action executed and reached on goal
        #     0: action executed
        #    -1: out of bounds
        #    -2: collision with obstacle

        # ACTION_COST, IDLE_COST, GOAL_REWARD, COLLISION_REWARD = -0.1, -0.2, 1.0, -1.0
        reward = ACTION_COST
        if action == 0: # staying still
            if self.world.getPos() == self.world.getGoal():
                reward = GOAL_REWARD
            else:
                reward = IDLE_COST
        else: # moving
            if (action_status == 1):   # reached goal
                reward = GOAL_REWARD
            elif (action_status == -2 or action_status == -1): # collision
                reward = COLLISION_REWARD

        self.finished |= (self.world.getPos() == self.world.getGoal())

        # Perform observation
        observation = self._observe()

        # next valid actions
        nextActions = self._listNextValidActions(action)

        return observation, reward, self.finished, nextActions

    def _listNextValidActions(self, prev_action=0):
#         available_actions = [0] # staying still always allowed
        available_actions = [] # staying still NOT allowed

        # Get current agent position
        ax, ay    = self.world.getPos()
        n_moves   = 9 if self.DIAGONAL_MOVEMENT else 5

        for action in range(1,n_moves):
            dx, dy    = self.world.getDir(action)
            if(ax+dx >= self.world.state.shape[0] or ax+dx < 0 or ay+dy >= self.world.state.shape[1] or ay+dy < 0): # Out of bounds
                continue
            if(self.world.state[ax+dx,ay+dy] < 0): # Collision
                continue

            #otherwise we are ok to carry out the action
            available_actions.append(action)

#        if opposite_actions[prev_action] in available_actions and len(available_actions) > 1:
#            available_actions.remove(opposite_actions[prev_action])
        if not available_actions:
            available_actions = [0]

        return available_actions

    
    ######## RENDERING STUFFS ########
    
    def drawStar(self, centerX, centerY, diameter, numPoints, color):
        outerRad=diameter//2
        innerRad=int(outerRad*3/8)
        #fill the center of the star
        angleBetween=2*math.pi/numPoints#angle between star points in radians
        for i in range(numPoints):
            #p1 and p3 are on the inner radius, and p2 is the point
            pointAngle=math.pi/2+i*angleBetween
            p1X=centerX+innerRad*math.cos(pointAngle-angleBetween/2)
            p1Y=centerY-innerRad*math.sin(pointAngle-angleBetween/2)
            p2X=centerX+outerRad*math.cos(pointAngle)
            p2Y=centerY-outerRad*math.sin(pointAngle)
            p3X=centerX+innerRad*math.cos(pointAngle+angleBetween/2)
            p3Y=centerY-innerRad*math.sin(pointAngle+angleBetween/2)
            #draw the triangle for each tip.
            poly=rendering.FilledPolygon([(p1X,p1Y),(p2X,p2Y),(p3X,p3Y)])
            poly.set_color(color[0],color[1],color[2])
            poly.add_attr(rendering.Transform())
            self.viewer.add_onetime(poly)

    def create_rectangle(self,x,y,width,height,fill,permanent=False):
        ps=[(x,y),((x+width),y),((x+width),(y+height)),(x,(y+height))]
        rect=rendering.FilledPolygon(ps)
        rect.set_color(fill[0],fill[1],fill[2])
        rect.add_attr(rendering.Transform())
        if permanent:
            self.viewer.add_geom(rect)
        else:
            self.viewer.add_onetime(rect)
    def create_circle(self,x,y,diameter,size,fill,resolution=20):
        c=(x+size/2,y+size/2)
        dr=math.pi*2/resolution
        ps=[]
        for i in range(resolution):
            x=c[0]+math.cos(i*dr)*diameter/2
            y=c[1]+math.sin(i*dr)*diameter/2
            ps.append((x,y))
        circ=rendering.FilledPolygon(ps)
        circ.set_color(fill[0],fill[1],fill[2])
        circ.add_attr(rendering.Transform())
        self.viewer.add_onetime(circ)

    def initColors(self):
        #c = {a+1:hsv_to_rgb(np.array([a/float(self.num_agents),1,1])) for a in range(self.num_agents)}
        return {1: np.asarray([1., 0., 0.])}

    def _render(self, mode='human', close=False, screen_width=800, screen_height=800, action_probs=None):
        if close == True:
            return

        # action_probs is an optional parameter which provides a visualization of the action probabilities of the agent at each step
        size=screen_width/max(self.world.state.shape[0],self.world.state.shape[1])
        colors=self.initColors()
        if self.viewer==None:
            self.viewer=rendering.Viewer(screen_width,screen_height)
            self.reset_renderer=True

        if self.reset_renderer:
            self.create_rectangle(0,0,screen_width,screen_height,(.6,.6,.6),permanent=True)
            for i in range(self.world.state.shape[0]):
                start=0
                end=1
                scanning=False
                write=False
                for j in range(self.world.state.shape[1]):
                    if(self.world.state[i,j]!=-1 and not scanning):#free
                        start=j
                        scanning=True
                    if((j==self.world.state.shape[1]-1 or self.world.state[i,j] == -1) and scanning):
                        end=j+1 if j==self.world.state.shape[1]-1 else j
                        scanning=False
                        write=True
                    if write:
                        x=i*size
                        y=start*size
                        self.create_rectangle(x,y,size,size*(end-start),(1,1,1),permanent=True)
                        write=False

        i,j=self.world.getPos()
        x=i*size
        y=j*size
        color=colors[self.world.state[i,j]]
        self.create_rectangle(x,y,size,size,color)
        i,j=self.world.getGoal()
        x=i*size
        y=j*size
        color=colors[self.world.goals[i,j]]
        self.create_circle(x,y,size,size,color)
        if self.world.getGoal() == self.world.getPos():
            color=(0,0,0)
            self.create_circle(x,y,size,size,color)

        if action_probs is not None:
            n_moves=9 if self.DIAGONAL_MOVEMENT else 5
            #take the a_dist from the given data and draw it on the frame
            a_dist = action_probs
            if a_dist is not None:
                for m in range(n_moves):
                    dx,dy=self.world.getDir(m)
                    x=(self.world.getPos()[0]+dx)*size
                    y=(self.world.getPos()[1]+dy)*size
                    s=a_dist[m]*size
                    self.create_circle(x,y,s,size,(0,0,0))

        self.reset_renderer=False
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

if __name__=='__main__':
    env = SAPPEnv(PROB=(.3,.5), SIZE=(10,11), DIAGONAL_MOVEMENT=False)
#    env.reset()

    print(env.world.state)
    print(env.world.getPos(), env.world.getGoal())
    print(env.world.goals)
    valid_actions = env._listNextValidActions()
    print(valid_actions)

    env.step(valid_actions[0])
    print(env.world.state)
    print(env.world.getPos())

    print(env._render())
