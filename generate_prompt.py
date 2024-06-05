import numpy as np
import pandas as pd
from prompt import *
from pathlib import Path

def read_all_command(path: str):
    commands_df = pd.read_csv(path, encoding = 'ISO-8859-1')
    commands_df.replace({'Yes': 1, 'No': 0, 'yes':1, 'no': 0}, inplace = True)
    preview = commands_df.head()

    all_commands_str = commands_df['Command'].str.lower().tolist()
    all_commands_ids = commands_df['Command ID'].tolist()
    task_name = commands_df.columns[2:].tolist()
    gt_array = commands_df[task_name].values
    return zip(all_commands_ids, all_commands_str), task_name, gt_array

def get_completion_from_user_input(
        user_input,
        provide_detailed_explain = False,
        provide_few_shots = False,
        step_by_step = False,
        num_shots = 4,
        batch = False,
    ):
    fix_system_message = system_message
    if step_by_step:
        fix_system_message = step_system_message
        
    messages =  [  
        {'role':'system', 'content': fix_system_message},
    ]

    if not step_by_step and provide_detailed_explain:
        messages.append({'role':'assistant', 'content': f'{delimiter}{assistant}{delimiter}'})

    if provide_few_shots:
        if num_shots == 4:
            messages.append({'role':'user', 'content': f'{delimiter}{few_shot_user_1}{delimiter}'})
            messages.append({'role':'assistant', 'content': few_shot_assistant_1})
            messages.append({'role':'user', 'content': f'{delimiter}{few_shot_user_2}{delimiter}'})
            messages.append({'role':'assistant', 'content': few_shot_assistant_2})
            messages.append({'role':'user', 'content': f'{delimiter}{few_shot_user_3}{delimiter}'})
            messages.append({'role':'assistant', 'content': few_shot_assistant_3})
            messages.append({'role':'user', 'content': f'{delimiter}{few_shot_user_4}{delimiter}'})
            messages.append({'role':'assistant', 'content': few_shot_assistant_4})
        elif num_shots == 3:
            messages.append({'role':'user', 'content': f'{delimiter}{few_shot_user_1}{delimiter}'})
            messages.append({'role':'assistant', 'content': few_shot_assistant_1})
            messages.append({'role':'user', 'content': f'{delimiter}{few_shot_user_2}{delimiter}'})
            messages.append({'role':'assistant', 'content': few_shot_assistant_2})
            messages.append({'role':'user', 'content': f'{delimiter}{few_shot_user_3}{delimiter}'})
            messages.append({'role':'assistant', 'content': few_shot_assistant_3})
        elif num_shots == 2:
            messages.append({'role':'user', 'content': f'{delimiter}{few_shot_user_1}{delimiter}'})
            messages.append({'role':'assistant', 'content': few_shot_assistant_1})
            messages.append({'role':'user', 'content': f'{delimiter}{few_shot_user_2}{delimiter}'})
            messages.append({'role':'assistant', 'content': few_shot_assistant_2})
        elif num_shots == 1:
            messages.append({'role':'user', 'content': f'{delimiter}{few_shot_user_1}{delimiter}'})
            messages.append({'role':'assistant', 'content': few_shot_assistant_1})

    # print('user_input:')
    # print(user_input)

    if not batch:
        messages.append({'role':'user', 'content': f'{delimiter}{user_input}{delimiter}'})
    else:
        for _ in user_input:
            messages.append({'role':'user', 'content': f'{delimiter} {_} {delimiter}\n'})

    # print(messages)

    reply_str = ''
    for message in messages:
        if message['role'] == 'system' or 'assistant' or 'user':
            reply_str += message['content']

    return reply_str

def split_list_to_n_sublists(input_list, n):
    """
    Splits a list into n approximately equal-sized sublists.

    :param input_list: The original list to split.
    :param n: The number of sublists.
    :return: A list of n sublists.
    """
    k, m = divmod(len(input_list), n)
    return [input_list[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]

if __name__ == '__main__':

    # Path( './prompt_input' ).mkdir( parents = True, exist_ok = True )

    Path( './prompt_input/ordinary' ).mkdir( parents = True, exist_ok = True )
    Path( './prompt_input/step_by_step' ).mkdir( parents = True, exist_ok = True )
    Path( './prompt_input/step_by_step_no_shot' ).mkdir( parents = True, exist_ok = True )
    Path( './prompt_input/explanation' ).mkdir( parents = True, exist_ok = True )

    provide_detailed_explain = True
    provide_few_shots = False
    step_by_step = True
    num_shots = 4
    batch = 10

    pd.set_option('future.no_silent_downcasting', True)
    commands_w_id, tasks, gt_array = read_all_command('./ucu_subset.csv')

    # batch_num = int(len(list(commands_w_id)) / batch)
    # print(f'batch_num: {batch_num}')

    i_list = []
    command_list = []

    commands_w_id
    for (i, command) in commands_w_id:
        # print(command)
        i_list.append(i)
        command_list.append(command)

    # for (i, command) in commands_w_id:
    #     print(f'i: {i}')
    #     print(f'command: {command}')

    #     response = get_completion_from_user_input(
    #         command, 
    #         provide_detailed_explain = provide_detailed_explain,
    #         provide_few_shots = provide_few_shots,
    #         step_by_step = step_by_step,
    #         num_shots = num_shots
    #     )
    #     print(response)
    #     break
    
    # print(f'command_list:')
    # print(command_list)

    batch_input_list = split_list_to_n_sublists( command_list, batch )
    for index, batch_input in enumerate(batch_input_list):
        # print(f'batch_input: {batch_input}')
        response = get_completion_from_user_input(
            batch_input, 
            provide_detailed_explain = provide_detailed_explain,
            provide_few_shots = provide_few_shots,
            step_by_step = step_by_step,
            num_shots = num_shots,
            batch = True
        )
        # print(response)

        if step_by_step and not provide_few_shots:
            kind = 'step_by_step_no_shot'
        elif step_by_step and provide_few_shots:
            kind = 'step_by_step'
        elif not step_by_step and provide_detailed_explain:
            kind = 'explanation'
        else:
            kind = 'ordinary'

        print(kind)

        with open( f'./prompt_input/{kind}/' + str(index + 1) + '.txt', mode = 'w+' ) as f:
            print(response, file = f)
