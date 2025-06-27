import transformers
from transformers import pipeline
from dyna_gym.pipelines import uct_for_dialogue_planning_pipeline
from dataset.durecdial import DuRecdial

# define a reward function based on sentiment of the generated text
sentiment_pipeline = pipeline("sentiment-analysis")
def reward_function(sentence, target):
    if target.lower() in sentence.lower():
        return 3.0
    else:
        return -3.0

if __name__ == '__main__':

    train_data_path = 'data/DuRecDial/data/en_train.txt'
    dev_data_path = 'data/DuRecDial/data/en_dev.txt'
    test_data_path = 'data/DuRecDial/data/en_test.txt'

    durecdial = DuRecdial(train_data_path=train_data_path,
                          dev_data_path= dev_data_path,
                          test_data_path= test_data_path
                          )

    # maximum number of steps / tokens to generate in each episode
    horizon = 10

    # arguments for the UCT agent
    uct_args = dict(
        action_space = durecdial.goals,
        rollouts = 20,
        gamma = 1.,
        width = 3,
        alg = 'uct', # or p_uct
    )

    pipeline = uct_for_dialogue_planning_pipeline(
        terminal_act="Say goodbye",
        horizon = horizon,
        action_space = durecdial.goals,
        reward_func = reward_function,
        uct_args = uct_args,
        should_plot_tree = True, # plot the tree after generation
    )

    initial_state = durecdial.test_instances[0]

    outputs = pipeline(initial_state=initial_state)
