import bc as bc
import torch
import numpy as np

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

def interact(env, learner, expert, observations, actions, validation_obs, validation_actions, checkpoint_path, seed, num_epochs=100, horizon=1000):
    """Interact with the environment and update the learner policy using DAgger.
   
    This function interacts with the given Gym environment and aggregates to
    the BC dataset by querying the expert.
    
    Parameters:
        env (Env)
            The gym environment (in this case, the Hopper gym environment)
        learner (Learner)
            A Learner object (policy)
        expert (ExpertActor)
            An ExpertActor object (expert policy)
        observations (torch.tensor or numpy.ndarray)
            A list of observations
        actions (torch.tensor or numpy.ndarray)
            A list of actions
        validation_obs (torch.tensor or numpy.ndarray)
            A list of validation observations
        validation_actions (torch.tensor or numpy.ndarray)
            A list of validation actions
        checkpoint_path (str)
            The path to save the best performing model checkpoint
        seed (int)
            The seed to use for the environment
        num_epochs (int)
            Number of epochs to run the train function for
    """
    # Ensure inputs are converted to tensors using from_numpy if necessary
    

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
                # Get the action from the learner
                learner_action = learner.get_action(obs)

                # Query the expert for the correct action
                expert_action = expert.get_expert_action(obs)

                # Take a step in the environment using the learner's action
                next_obs, reward, done, _ = env.step(learner_action)

                # Append the current observation and expert action to the episode data
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
        bc.train(learner, observations, actions, validation_obs, validation_actions, None, num_epochs)

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