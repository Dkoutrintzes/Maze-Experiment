from rl_models.sac_discrete_agent import DiscreteSACAgent


def get_sac_agent(args,config, env, chkpt_dir=None,p_name="None",ID='First'):

    mode = config['Experiment']['mode']
    
    buffer_max_size = config['Experiment'][mode]['buffer_memory_size']
    update_interval = config['Experiment'][mode]['train_games_per_block']
    scale = config['Experiment'][mode]['reward_scale']
    
    if config['game']['agent_only']:
        # up: 1, down:2, left:3, right:4, upleft:5, upright:6, downleft: 7, downright:8
        action_dim = pow(2, env.action_space.actions_number)
    else:
        action_dim = env.action_space.actions_number
    print("action_dim", action_dim)
    print('Creating agent: ',ID)
    sac = DiscreteSACAgent(args=args,config=config, env=env,
                            n_actions=action_dim,
                            chkpt_dir=chkpt_dir, buffer_max_size=buffer_max_size, update_interval=update_interval,
                            reward_scale=scale,participant_name=p_name,ID=ID)

    return sac

