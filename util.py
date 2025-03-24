import cv2, math
from matplotlib.colors import hsv_to_rgb
#import imageio
import imageio.v3 as imageio
import numpy as np

def make_gif(images, file_name):
    """record gif"""
    #imageio.mimwrite(file_name, images, subrectangles=True)
    imageio.imwrite(file_name, images, duration=100, loop=0)
    print("wrote gif")

def init_colors():
    """the colors of agents and goals"""
    c = {}
    c[0] = [1,1,1]
    c[-1] = [0,0,0]
    c[-2] = [0.5,0.5,0.5]
    c[1] = [0,0,1]
    
    return c

def getRectPoints(coord, scale):
    base = [coord[1]*scale, coord[0]*scale]
    return np.array([base, [base[0]+scale-1, base[1]], [base[0]+scale-1,base[1]+scale-1], [base[0], base[1]+scale-1]])    

def pixelForText(coord, scale):
    base = [coord[1]*scale, coord[0]*scale]
    return [int(math.floor(base[0]+scale*1/4)), int(math.floor(base[1]+scale*3/4))]


def getCenter(coord, scale):
    base = [coord[1]*scale, coord[0]*scale]
    return [int(math.floor(base[0]+scale/2)), int(math.floor(base[1]+scale/2))]

def getTriPoints( coord, scale):
    base = [coord[1]*scale, coord[0]*scale]
    return  np.array([[int(math.floor(base[0]+scale/2)), base[1]], [base[0]+scale-1,base[1]+scale-1], [base[0], base[1]+scale-1]])    

def renderWorld(world = np.zeros(1), scale=50, agent=None, goal=None):
    size = world.shape
    screen_height = scale*size[0]
    screen_width = scale*size[1]

    colours = init_colors()

    scene = np.zeros([screen_height, screen_width, 3])

    for coord,val in np.ndenumerate(world):
        cv2.fillPoly(scene, pts=[getRectPoints(coord=coord, scale=scale)], color=colours[val])


    cv2.circle(scene, getCenter(coord=goal, scale=scale), math.floor(scale/2)-1, colours[1], -1)
        # cv2.putText(scene, str(val+1), pixelForText(goal, scale), cv2.FONT_HERSHEY_SIMPLEX,scale/40, (0,0,0), int(scale/20))

    cv2.fillPoly(scene, pts=[getTriPoints(coord=agent, scale=scale)], color=colours[1])
    
        # cv2.putText(scene, str(val+1), pixelForText(agent, scale), cv2.FONT_HERSHEY_SIMPLEX,scale/40, (0,0,0), int(scale/20))
    scene = scene*255
    scene = scene.astype(dtype='uint8')
    return scene