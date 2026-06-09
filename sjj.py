from fairino import Robot
import time
IP='192.168.40.129'
robot=Robot.RPC(IP)

def init():
    home_pose=(0,-90,90,0,90,0)
    robot.SetSpeed(50)
    robot.MoveJ(home_pose, 0, 0)
    return True

#rx上下抬头，ry水平旋转 rz左右翻转

if __name__ == "__main__":
    print("初始化成功" if init() else "初始化失败")
    init()
    err,pos=robot.GetActualTCPPose()
    pos[3]+=55
    #pos[0]+=50
    robot.MoveCart(pos,0,0)
    pos[2]+=80
    robot.MoveCart(pos,0,0)
        