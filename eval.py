import numpy as np
import pandas as pd
import re

def read_all_command(path: str):
    commands_df = pd.read_csv(path, encoding = 'ISO-8859-1')
    commands_df.replace({'Yes': 1, 'No': 0, 'yes':1, 'no': 0}, inplace = True)
    preview = commands_df.head()

    all_commands_str = commands_df['Command'].str.lower().tolist()
    all_commands_ids = commands_df['Command ID'].tolist()
    task_name = commands_df.columns[2:].tolist()
    gt_array = commands_df[task_name].values
    return zip(all_commands_ids, all_commands_str), task_name, gt_array

def extract_outputs(text, i = -1):
    
    pattern = r"Output is //\[([0-1\s]+)\]//"
    matches = re.findall(pattern, str(text))

    # print(f'matches: {matches}')

    if len(matches) == 0:
        pattern = r"//\[([0-1\s]+)\]//"
        matches = re.findall(pattern, str(text))
    if len(matches) == 0:
        pattern = r"would be //\[([0-1\s]+)\]//"
        matches = re.findall(pattern, str(text))
    if len(matches) == 0:
        print(f"No match found in {bc.FAIL}{text}{bc.ENDC}, Need check later at {i}.")
        return np.array([-1 for _ in range(8)])
    
    # np_res = [np.fromstring(match, dtype = int, sep = ' ') for match in matches][0]
    # if np_res.shape[0] != 8:
    #     print(f"Wrong shape in {text}, Need check later at {i}.")
    #     return np.array([-1 for _ in range(8)])
    # return np_res

    np_res = np.asarray([np.fromstring(match, dtype = int, sep = ' ') for match in matches])
    # print(np_res)

    for ans in np_res:
        if ans.shape[0] != 8:
            print(f"Wrong shape in {text}, Need check later at {i}.")
            return np.array([-1 for _ in range(8)])
    return np_res

def get_acc_single_col_acc( predict_result, ground_truth ):

    if predict_result.shape != ground_truth.shape:
        raise Exception( 'Shape should be same!' )

    res = []
    for i in range(predict_result.shape[1]):
        _ = (predict_result[:, i] == ground_truth[:, i])
        res.append( np.count_nonzero(_) / predict_result.shape[0] )

    return np.asarray(res)

def get_all_acc( predict_result, ground_truth ):
    
    if predict_result.shape != ground_truth.shape:
        raise Exception( 'Shape should be same!' )
    
    _ = (predict_result == ground_truth)
    acc = np.count_nonzero(_) / np.prod(predict_result.shape)
    return acc

def get_com_acc( predict_result, ground_truth ):
    
    if predict_result.shape != ground_truth.shape:
        raise Exception( 'Shape should be same!' )
    
    _ = 0
    for y_1, y_2 in zip( predict_result, ground_truth ):
        if np.sum(y_1 == y_2) == predict_result.shape[1]:
            _ += 1

    acc = _ / predict_result.shape[0]
    return acc

if __name__ == '__main__':
    pd.set_option('future.no_silent_downcasting', True)
    commands_w_id, tasks, gt_array = read_all_command('./ucu_subset.csv')

    model = 'GPT4o'
    # model = 'GPT4'
    # model = 'GPT3.5'
    # model = 'Claude_Sonet'

    # kind = 'step_by_step'
    # kind = 'explaination'
    # kind = 'ordinary'
    kind = 'step_by_step_no_shot'

    chat_record_path = f'./chat_result/{model}/{kind}'

    init_flag = True
    all_pred = None

    for i in range(10):
        with open(chat_record_path + f'/{i + 1}.txt') as f:
            text = f.read()
            res = extract_outputs(text)
            if init_flag:
                all_pred = res
                init_flag = False
            else:
                all_pred = np.vstack( (all_pred, res) )
    
    print('all_pred:')
    print(all_pred)
    print(all_pred.shape)

    acc_1 = get_all_acc( all_pred, gt_array )
    acc_2 = get_com_acc( all_pred, gt_array )
    acc_3 = get_acc_single_col_acc( all_pred, gt_array )
    
    print(f'pointwise_acc: {acc_1}')
    print(f'command_acc: {acc_2}')
    print('task_array:')
    print(tasks)
    print(f'tasks_acc: {acc_3}')

    with open( chat_record_path + '/eval_result.txt', mode = 'w+' ) as f:
        print(f'pointwise_acc: {acc_1}', file = f)
        print(f'command_acc: {acc_2}', file = f)
        print('task_array:', file = f)
        print(tasks, file = f)
        print(f'tasks_acc: {acc_3}', file = f)


