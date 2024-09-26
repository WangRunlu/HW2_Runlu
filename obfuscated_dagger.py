import gym
import torch
import numpy as np
import bc as bc

class PartialObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(PartialObservationWrapper, self).__init__(env)
        # Modify the observation space to reflect the reduced dimensionality
        original_shape = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low[:-1],
            high=env.observation_space.high[:-1],
            dtype=env.observation_space.dtype
        )
        self.hidden_index = original_shape - 1  # Index of the hidden observation

    def observation(self, observation):
        # Return the observation without the last index
        return observation[:-1]

def evaluate(env, learner):
    NUM_TRAJS = 50
    total_learner_reward = 0
    for i in range(NUM_TRAJS):
        done = False
        obs = env.reset(seed=i)
        while not done:
            with torch.no_grad():
                action = learner.get_action(obs)
            obs, reward, done, _ = env.step(action)
            total_learner_reward += reward
            if done:
                break
    return total_learner_reward / NUM_TRAJS

def interact(
    env,
    learner,
    expert,
    observations,
    actions,
    validation_obs,
    validation_actions,
    checkpoint_path,
    seed,
    num_epochs=100,
    horizon=1000,
):
    """Interact with the environment and update the learner policy using DAgger.

    This function interacts with the given Gym environment and aggregates to
    the BC dataset by querying the expert.

    Parameters:
        env (Env): The gym environment (partially observable)
        learner (Learner): A Learner object (policy)
        expert (ExpertActor): An ExpertActor object (expert policy)
        observations (torch.tensor or numpy.ndarray): A list of observations
        actions (torch.tensor or numpy.ndarray): A list of actions
        validation_obs (torch.tensor or numpy.ndarray): A list of validation observations
        validation_actions (torch.tensor or numpy.ndarray): A list of validation actions
        checkpoint_path (str): The path to save the best performing model checkpoint
        seed (int): The seed to use for the environment
        num_epochs (int): Number of epochs to run the train function for
    """
    NUM_INTERACTIONS = 5
    best_reward = 0
    best_model_state = None

    for episode in range(NUM_INTERACTIONS):
        # Aggregate 10 trajectories per interaction
        for _ in range(10):
            done = False
            obs = env.reset(seed=seed)
            episode_obs = []
            episode_actions = []

            for _ in range(horizon):
                # Get the action from the learner using partial observation
                learner_action = learner.get_action(obs)

                # Get the full observation for the expert
                full_obs = env.unwrapped._get_obs()

                # Query the expert for the correct action using full observation
                expert_action = expert.get_expert_action(full_obs)

                # Take a step in the environment using the learner's action
                next_obs, reward, done, _ = env.step(learner_action)

                # Append the current partial observation and expert action
                episode_obs.append(obs)
                episode_actions.append(expert_action)

                # Update the observation to the next state
                obs = next_obs

                if done:
                    break

            # Convert episode observations and actions to tensors and cast to float
            episode_obs_tensor = torch.tensor(episode_obs).float()
            episode_actions_tensor = torch.tensor(episode_actions).float()

            # Concatenate the current episode's data with the existing observations and actions
            observations = torch.cat((observations, episode_obs_tensor), dim=0)
            actions = torch.cat((actions, episode_actions_tensor), dim=0)

        # Train the learner using the aggregated dataset (including the new trajectories)
        bc.train(
            learner,
            observations,
            actions,
            validation_obs,
            validation_actions,
            None,
            num_epochs,
        )

        # Evaluate the learner's performance after the update
        reward = evaluate(env, learner)
        print(f"After interaction {episode+1}, reward = {reward}")

        # Save the model if the current reward exceeds the best reward
        if reward > best_reward:
            best_reward = reward
            best_model_state = learner.state_dict()

    # Save the best performing checkpoint
    if best_model_state is not None:
        torch.save(best_model_state, checkpoint_path)

# Usage Example
if __name__ == "__main__":
    import expert_policy  # Assuming you have an expert policy module
    from learner import Learner  # Your learner implementation

    # Create the original Hopper environment
    env = gym.make("Hopper-v2")

    # Wrap the environment to make it partially observable
    env = PartialObservationWrapper(env)

    # Initialize the expert and learner
    expert = expert_policy.ExpertActor()
    learner = Learner()

    # Collect initial BC dataset
    observations, actions = collect_bc_data(expert, env)
    validation_obs, validation_actions = observations[:100], actions[:100]
    observations, actions = observations[100:], actions[100:]

    # Train initial policy using BC
    bc.train(
        learner,
        observations,
        actions,
        validation_obs,
        validation_actions,
        None,
        num_epochs=100,
    )

    # Run DAgger
    interact(
        env,
        learner,
        expert,
        observations,
        actions,
        validation_obs,
        validation_actions,
        checkpoint_path="best_model.pth",
        seed=42,
        num_epochs=100,
    )
