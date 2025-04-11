import argparse
import os
import sys
from search.mcts_locomotion_task import MCTSPhaseLocomotionTask
from search.utils.save import load_phase_sequence_from_yaml

def import_from_run_dir(run_dir, module_name):
    import importlib.util

    # Add the directory to the system path
    sys.path.insert(0, run_dir)
    try:
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(run_dir, f"{module_name}.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Extract the required variables
        mpc_solver = getattr(module, 'mpc_solver', None)
        mpc_close_loop = getattr(module, 'mpc_close_loop', None)
        sim = getattr(module, 'sim', None)
        surfaces = getattr(module, 'surfaces', None)

        if mpc_solver is None or mpc_close_loop is None or sim is None or surfaces is None:
            raise AttributeError("One or more required variables (mpc_solver, mpc_close_loop, sim) are missing in the module.")

        return mpc_solver, mpc_close_loop, sim, surfaces
    finally:
        # Clean up the system path
        sys.path.pop(0)
        
def main():
    parser = argparse.ArgumentParser(description="Process a directory path.")
    parser.add_argument('run_dir', type=str, help='Path to the search directory')
    parser.add_argument('it', type=int, help='Solution iteration number to replay')
    args = parser.parse_args()

    run_dir = args.run_dir
    it = args.it
    solution_dir = os.path.join(run_dir, f"iteration_{it}")
    
    if os.path.isdir(run_dir) and os.path.isdir(solution_dir):
        print(f"Searching in directory: {run_dir}")
        print(f"Solution dir for iteration {it} found.")    

        try:
            file_name = next(f for f in os.listdir(run_dir) if f.endswith('.py'))
            module_name = file_name[:-3]  # Remove the .py extension
            mpc_solver, mpc_close_loop, sim, surfaces = import_from_run_dir(run_dir, module_name)
            mcts = MCTSPhaseLocomotionTask(
                C=1,
                alpha_exploration=1,
                sim=sim,
                mpc_solver=mpc_solver,
                mpc_close_loop=mpc_close_loop,
                n_phases=1,
                surfaces=surfaces,
                goal_surf_id=[],
            )
            node_sequence, nodes_per_phase = load_phase_sequence_from_yaml(solution_dir)
            success = mcts.run_mpc(node_sequence, nodes_per_phase, record_video=False, use_viewer=True)          
            print(sim.mj_data.qpos[2])
            print("Success", success)
            
        except StopIteration:
            print("Error: No Python file found in the directory.")
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print(f"Error: {run_dir} is not a valid directory.")

if __name__ == "__main__":
    main()