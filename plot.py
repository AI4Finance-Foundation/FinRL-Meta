import matplotlib.pyplot as plt
import numpy as np

def draw_cumulative_return(episode_returns: np.array, save_dir: str):
    plt.plot(episode_returns, label='agent return')
    plt.grid()
    plt.title('cumulative return')
    plt.xlabel('time')
    plt.legend()
    plt.savefig(f'{save_dir}/cumulative_return.jpg')
    
