from matplotlib import pyplot as plt
import pandas as pd


def rew_mean():
    df = pd.read_csv("./data/PPO_Mean.csv")
    plt.plot(df.Step, df.Value)
    plt.ylabel("Reward Mean per Episode")
    plt.xlabel("Timesteps")
    plt.title("PPO")
    plt.show()


def fpsgraphs():
    df = pd.read_csv("./data/PPO_FPS.csv")
    df2 = pd.read_csv("./data/PPO_CNN_FPS.csv")
    df3 = pd.read_csv("./data/A2C_FPS.csv")
    df4 = pd.read_csv("./data/A2C_CNN_FPS.csv")
    plt.plot(df.Step, df.Value, label = 'PPO', color =  'r')
    plt.plot(df2.Step, df2.Value, color='orange', label = 'PPO_CNN'), 
    plt.plot(df3.Step, df3.Value, 'b', label = 'A2C')
    plt.plot(df4.Step, df4.Value, 'c', label = 'A2C_CNN')
    plt.xlabel("Timesteps")
    plt.ylabel("FPS")
    plt.title("FPS Over Timesteps")
    plt.legend()
    plt.show()

    df.columns = [c.replace(' ', '_') for c in df.columns]
    df2.columns = [c.replace(' ', '_') for c in df2.columns]
    df3.columns = [c.replace(' ', '_') for c in df3.columns]
    df4.columns = [c.replace(' ', '_') for c in df4.columns]
    PPO = (df['Wall_time'].values[-1:] - df['Wall_time'].values[0:])/3600
    PPO_CNN = (df2['Wall_time'].values[-1:] - df2['Wall_time'].values[0:])/3600
    A2C = (df3['Wall_time'].values[-1:] - df3['Wall_time'].values[0:])/3600
    A2C_CNN = (df4['Wall_time'].values[-1:] - df4['Wall_time'].values[0:])/3600


    Y = [PPO[0], PPO_CNN[0], A2C[0], A2C_CNN[0]]
    y_axis = []
    for time in Y:
        
        w = str("{:0.2f}".format(time))
        
        y_axis.append(float(w))
    x_axis = ['PPO', 'PPO_CNN', 'A2C', "A2C_CNN" ]
    plt.bar(x_axis, y_axis, color=['r', 'orange', 'blue', 'cyan'])
    plt.ylabel("Time in Hours")
    plt.xlabel("Models")
    plt.show()


rew_mean()
fpsgraphs()