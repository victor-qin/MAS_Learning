import PDP
import JinEnv
import numpy as np


# --------------------------- load environment ----------------------------------------
uav = JinEnv.Quadrotor()
Jx, Jy, Jz, mass, win_len = 1, 1, 1, 1, 0.4
uav.initDyn(Jx=Jx, Jy=Jy, Jz=Jz, mass=mass, l=win_len, c=0.01)
wr, wv, wq, ww = 1, 1, 5, 1
uav.initCost(wr=wr, wv=wv, wq=wq, ww=ww, wthrust=0.1)

uav.play_animation(win_len, [(0,0,0)])


print("done")