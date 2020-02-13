import tensorflow as tf
import numpy as np
import gym
import yaml
import os

from helpers import discount_rewards
from models import *

def train_model(num_episodes, model_version, discount_rate, learning_rate):
    """
    Trains a policy model

    :param num_epochs: int. Determines how many epochs to train for.
    :param model_version: str. Provides the model architecture to use.
    :param discount_rate: float. Determines the impact of future actions on current reward.
      Valid entries in (0, 1) where a larger discount_rate forces model to consider current actions
      to have smaller effects on future rewards.
    :param learning_rate: float. Determines how far of a step is taken along gradient when training.
    :return: None
    """

    # getting available models
    with open('model_architectures.yaml', 'r') as file:
        available_models = yaml.safe_load(stream = file)
        file.close()

     # input checks
    if type(num_episodes) != int:
        raise TypeError('num_episodes must be of type int')
    if num_episodes <= 0:
        raise ValueError('num_episodes must be greater than zero')
    if type(model_version) != str:
        raise TypeError('model_version must be of type string')
    if model_version not in available_models:
        raise ValueError('model_version not available')
    if type(discount_rate) != int and type(discount_rate) != float:
        raise TypeError('discount rate must be of type int or float')
    if discount_rate <= 0:
        raise ValueError('discount_rate must be greater than zero')
    if type(learning_rate) != float:
        raise TypeError('learning_rate must be of type float')
    if learning_rate <= 0 or learning_rate >= 1:
        raise ValueError('learning_rate must be within (0, 1)')

    # determining model sub-version
    model_sub_version_write = False
    sub_version = 0
    while not model_sub_version_write:
        if not os.path.exists(os.path.join('trained_models', model_version, model_version + '.' + str(sub_version) + '.0')):
            model_sub_version_write = True
            break
        sub_version += 1

    # building model
    model = eval(available_models[model_version])()
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    # holding gradients
    gradient_holder = model.trainable_variables
    for i, gradient in enumerate(gradient_holder):
        gradient_holder[i] = gradient * 0

    # creating gym environment
    env = gym.make('CartPole-v1')
    env._max_episode_steps = 15000

    scores = []
    every_update = 5

    # training loop
    for episode in range(num_episodes + 1):
        observation = env.reset()

        episode_memory = []
        episode_score = 0
        done = False

        while not done:

            # creates vector of form [Position, Velocity, Angle, Angular Velocity]
            observation = observation.reshape([1, 4])

            with tf.GradientTape() as tape:

                # creating loss function and action
                logits = model(observation)
                a_dist = logits.numpy()
                action = np.random.choice(a = a_dist[0], p = a_dist[0])
                action = np.argmax(a_dist == action)
                loss = compute_loss([action], logits)

            # performing action and getting feedback from environment
            observation, reward, done, info = env.step(action)

            episode_score += reward

            # trick for quicker convergence
            if done:
                reward -= 10

            # getting gradients
            gradients = tape.gradient(target = loss, sources = model.trainable_variables)
            episode_memory.append([gradients, reward])

        scores.append(episode_score)

        # discounting rewards
        episode_memory = np.array(episode_memory)
        episode_memory[:, 1] = discount_rewards(rewards = episode_memory[:, 1], discount_rate = discount_rate)

        # applying rewards to corresponding gradients
        for grads, reward in episode_memory:
            for i, grad in enumerate(grads):
                gradient_holder[i] += grad * reward

        # back-propagating gradients, resetting gradients
        if episode % every_update == 0:
            optimizer.apply_gradients(zip(gradient_holder, model.trainable_variables))
            for i, grad in enumerate(gradient_holder):
                gradient_holder[i] = grad * 0

        if episode % 100 == 0:
            print('Episode {} Score {}'.format(episode, np.mean(scores[-20:])))
            tf.keras.models.save_model(model = model,
                                       filepath = os.path.join('trained_models', model_version, model_version + '.' +
                                                             str(sub_version) + '.{}'.format(episode)))

    final_performance = int(round(np.mean(scores[-20:])))

    # dumping training results into yaml file
    yaml_dump = {}
    yaml_dump['Model Version'] = model_version
    yaml_dump['Model Sub-version'] = sub_version
    yaml_dump['Number of Training Episodes'] = num_episodes
    yaml_dump['Discount Rate'] = discount_rate
    yaml_dump['Learning Rate'] = learning_rate
    yaml_dump['Final Performance'] = final_performance

    with open(os.path.join('trained_models', model_version, model_version + '.' + str(sub_version) + '_training_details.yaml'), 'w') as file:
        yaml.dump(data = yaml_dump, stream = file)
        file.close()


if __name__ == '__main__':

    num_episodes = 1000
    model_version = 'v3'
    discount_rate = 0.8
    learning_rate = 0.01

    train_model(num_episodes = num_episodes, model_version = model_version,
                discount_rate = discount_rate, learning_rate = learning_rate)
