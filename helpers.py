import numpy as np

def discount_rewards(rewards, discount_rate = 0.8):
  """
  Takes in rewards and applies discount_rate

  Reward = Reward[t=0] * pow(discount_rate, 0) + ... + Reward[t=n] * pow(discount_rate, n)

  :param rewards: numpy.array. The list of rewards to be discounted.
  :param discount_rate: float. Determines the impact of future actions on current reward.
      Valid entries in (0, 1) where a larger discount_rate forces model to consider current actions
      to have smaller effects on future rewards.
  :return: discounted_rewards: numpy.ndarray. Discounted rewards.
  """

  # input checks
  if type(rewards) != np.ndarray:
    raise TypeError('rewards must be of type numpy.array')
  if type(discount_rate) != float:
    raise TypeError('discount_rate must be of type float')
  if not 0 < discount_rate < 1:
    raise ValueError('discount_rate must be in (0, 1)')

  # applying discount in fashion described above
  discounted_rewards = np.zeros_like(a = rewards)
  running_total = 0
  for i in reversed(range(0, rewards.size)):
    running_total = running_total * discount_rate + rewards[i]
    discounted_rewards[i] = running_total
  return discounted_rewards