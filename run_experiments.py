import subprocess
import time
import os

def modify_param_script(file_path : str, parameter : str, new_value ):
    """
    Modifies the line containing HEIGHT= in the specified file.

    Args:
        file_path (str): Path to the file to modify.
        new_height (float): The new height value to set.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        with open(file_path, 'w') as file:
            for line in lines:
                if line.strip().startswith(f"{parameter} = "):
                    if isinstance(new_value, str):
                        file.write(f"{parameter} = '{new_value}'\n")
                    else:
                        file.write(f"{parameter} = {new_value}\n")
                        
                else:
                    file.write(line)
    except Exception as e:
        print(f"An error occurred: {e}")

def run_script(script_path: str):
    """
    Runs the specified Python script.

    Args:
        script_path (str): Path to the Python script to run.
    """
    try:
        result = subprocess.run(['python3', script_path], check=True, capture_output=True, text=True)
        print("Script output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Script failed with error:\n{e.stderr}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    config_file_path = "/home/atari/workspace/configs_mpc_solver.py"
    exp_dir = "experiments/binary_reward"
    N_repeat = 3
    
    task_name = "climb_box"
    script_path = f"/home/atari/workspace/search_{task_name}_phase.py"
    heights = [0.2, 0.3, 0.4, 0.5, 0.6]
    heights = [0.2, 0.4, 0.5, 0.6]
    parameter_name = "HEIGHT"
    for height in heights:
        base_save_dir = os.path.join(exp_dir, task_name, parameter_name.lower(), str(height))
        os.makedirs(base_save_dir, exist_ok=True)
        modify_param_script(config_file_path, "BASE_SAVE_DIR", base_save_dir)
        
        for _ in range(N_repeat):
            modify_param_script(script_path, parameter_name, height)
            run_script(script_path)
            time.sleep(2)
            
            
    task_name = "cross_gap"
    script_path = f"/home/atari/workspace/search_{task_name}_phase.py"
    lengths = [0.3, 0.5, 0.7, 0.9, 1.1]
    parameter_name = "GAP_LENGTH"
    for length in lengths:
        base_save_dir = os.path.join(exp_dir, task_name, parameter_name.lower(), str(length))
        os.makedirs(base_save_dir, exist_ok=True)
        modify_param_script(config_file_path, "BASE_SAVE_DIR", base_save_dir)
        
        for _ in range(N_repeat):
            modify_param_script(script_path, parameter_name, length)
            run_script(script_path)
            time.sleep(2)