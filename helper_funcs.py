from libs import *

from stable_baselines3.common.monitor import Monitor

from callback import SaveOnBestTrainingRewardCallback


def read_data(cc):
    data_path = f'data\\{cc}USD_1h.csv'
    df = pd.read_csv(data_path, skiprows=1)
    return df


def reformat_data(df, cc):
    print('NaNs in our data:\n')
    print(df.isna().sum())
    
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True) 
    df = df[['open', 'high', 'low', 'close', f'Volume {cc}']]
    df = df.rename(columns = {f'Volume {cc}':'Volume',
                               'open': 'Open',
                               'high': 'High',
                               'low': 'Low',
                               'close': 'Close'})
    
    print('\nThe reformatted data:')
    print(df.head(2))
    return df


def prepare_traintest_data(df, train_len):
    train_df = df.iloc[: int(train_len * len(df)), :]
    test_df = df.iloc[int(train_len * len(df)) :, :]
    print(f'Length of training data: {len(train_df)}\n')
    print(f'Length of test data: {len(test_df)}')
    
    return train_df, test_df


def create_dummy_env(cc, df, window_size, log_dir):
    env_maker = gym.make('stocks-v0', 
                         df=df, 
                         frame_bound=(window_size, len(df)), 
                         window_size=window_size) 
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env_maker, log_dir)
    env = DummyVecEnv([lambda: env_maker])
    
    f = open(f"results_{cc}.txt", "a")
    f.write(f'\nWindow size: {window_size}\n')
    f.close()
    
    callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir, verbose=0)
    
    return env, callback


def create_model(cc, env, policy_type, policy_kwargs, comb, verbose, total_timesteps, callback):
    if policy_type == 'A2C':
        model = A2C('MlpPolicy', env, verbose=verbose, policy_kwargs=policy_kwargs) 
        model.learn(total_timesteps=total_timesteps, callback=callback)
    else:
        model = PPO('MlpPolicy', env, verbose=verbose, policy_kwargs=policy_kwargs) 
        model.learn(total_timesteps=total_timesteps, callback=callback)
        
    f = open(f"results_{cc}.txt", "a")
    f.write(f'Policy: {policy_type}\n')
    f.write(f'Policy kwargs: {comb}\n')
    f.write(f'Timesteps: {total_timesteps}\n')
    f.close()
    
    return model


def evaluate_model(cc, df, window_size, model):
    env = gym.make('stocks-v0', 
                   df=df, 
                   frame_bound=(window_size, len(df)), 
                   window_size=window_size)
    obs = env.reset()
    
    f = open(f"results_{cc}.txt", "a")
    while True: 
        obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            print("performance:", info)
            f.write(f"performance: {info}\n")
            f.write(f"max_possible_profit: {env.max_possible_profit()}")
            break
    
    f.close()
    
    
def test_hp_set(cc, train_df, test_df, window_size, log_dir, comb, policy, verbose, timesteps):
    env, callback = create_dummy_env(cc, train_df, window_size, log_dir)
    
    if len(comb) == 2:
        policy_kwargs = dict(net_arch=[dict(vf=comb[0], pi=comb[1])], activation_fn=ReLU)
    else:
        policy_kwargs = dict(net_arch=[comb[0], dict(vf=comb[1], pi=comb[2])], activation_fn=ReLU)
    model = create_model(cc, env, policy, policy_kwargs, comb, verbose, timesteps, callback)

    evaluate_model(cc, test_df, window_size, model)
    
def analyze_results(cc, profit_bound):
    best_models = []
    with open(f'results_{cc}.txt', 'r') as file:
        f = file.readlines()
    for (i, line) in enumerate(f):
        if 'total_profit' in line: 
            total_profit = float(line.split("'total_profit': ")[1].split(',')[0])
            if total_profit > profit_bound:
                window_size = int(f[i - 4].split('Window size: ')[1])
                policy = f[i - 3].split('Policy: ')[1].replace('\n', '')
                arg_els = f[i - 2].split('Policy kwargs: ')
                if arg_els[1].count(',') == 1:
                    if '64' in arg_els[1]:
                        shared, vf, pg = '/', 64, 64
                    else:
                        shared, vf, pg = '/', 128, 128
                else:
                    shared, vf, pg = 64, 64, 64
                timesteps = f[i - 1].split('Timesteps: ')[1].replace('\n', '')
                temp_dict = {'window size': window_size,
                             'policy':      policy,
                             'shared layers': shared,
                             'value network': vf,
                             'policy network': pg,
                             'timesteps': timesteps,
                             'total_profit': total_profit}

                best_models.append(temp_dict)
            
    result_df = pd.concat([pd.Series(model) for model in best_models], axis=1).T
    result_df = result_df.drop_duplicates()
    
    return result_df
