import tensorflow as tf
import numpy as np
import gym
import os

def run(model_version):
    """
    Runs an instance of OpenAI Gym's Cartpole-v1 environment and uses a trained policy model to balance the pole

    :param model_version: str. Specifies which trained model to use.
    :return: None
    """

    # input checks
    if type(model_version) != str:
        raise TypeError('model_version must be of type string')
    if not os.path.exists(os.path.join('trained_models', model_version[0: 2],model_version)):
        raise IOError('this model does not exist')

    # loading model and environment
    model = tf.keras.models.load_model(filepath = os.path.join('trained_models', model_version[0: 2], model_version))
    env = gym.make('CartPole-v1')
    env._max_episode_steps = 15000

    done = False
    observation = env.reset()
    rounds_lasted = 0

    while not done:
        env.render()
        action = model.predict(observation.reshape([1, 4]))
        action = np.argmax(action)
        observation, reward, done, info = env.step(action)

        rounds_lasted += 1

    env.close()
    print('Lasted for {} rounds'.format(rounds_lasted))

if __name__ == '__main__':

    model_version = 'v5.0.900'
    run(model_version = model_version)

