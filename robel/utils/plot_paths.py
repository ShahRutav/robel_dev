import pickle
from vtils.plotting.simple_plot import plot, show_plot, save_plot
import argparse

def plot_path(path, fig_name, info_keys=None):
  
    # obs / user
    if info_keys:
        for key in sorted(info_keys):
            print(key)
            plot(path['infos'][key], subplot_id=(2,3,1), legend=key,  plot_name='user', fig_name=fig_name)
    else:
        plot(path['observations'], subplot_id=(2,3,1),  plot_name='observations', fig_name=fig_name)
    # actions
    plot(path['actions'], subplot_id=(2,3,4),  plot_name='actions', fig_name=fig_name)

    # score 
    score_keys = [key for key in path['infos'].keys() if 'score' in key]
    for key in sorted(score_keys):
        plot(path['infos'][key], subplot_id=(1,3,3), legend=key[6:],  plot_name='score', fig_name=fig_name)

    # Rewards
    rewards_keys = [key for key in path['infos'].keys() if 'reward' in key] 
    for key in sorted(rewards_keys):
        plot(path['infos'][key], subplot_id=(1,3,2), legend=key[7:], plot_name='rewards', fig_name=fig_name)
 
    


# Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description="Plots rollouts")

    parser.add_argument("-p", "--paths", 
                        type=str, 
                        help="path to rollout-paths",
                        default="paths.pkl")
    parser.add_argument("-s", "--save_path", 
                        type=str, 
                        help="path to save-plots",
                        default=None)
    parser.add_argument("-i", "--info_keys", 
                        action='append', 
                        help="user info keys",
                        default=None)
    return parser.parse_args()

if __name__ == '__main__':
    
    # get args
    args = get_args()
    print(args.paths)
    paths = pickle.load(open(args.paths, 'rb'))

    for i, path in enumerate(paths):
        plot_path(path, fig_name="path{}.pdf".format(i), info_keys=args.info_keys)
        if args.save_path is not None:
            save_plot(args.save_path)
        else:
            show_plot()
            

