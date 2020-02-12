import numpy as np

def discount_rewards(rewards, discount_rate = 0.8):
  """

  :param rewards:
  :param discount_rate:
  :return:
  """

  discounted_rewards = np.zeros_like(a = rewards)
  running_total = 0
  for i in reversed(range(0, rewards.size)):
    running_total = running_total * discount_rate + rewards[i]
    discounted_rewards[i] = running_total
  return discounted_rewards