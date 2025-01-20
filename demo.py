import os
import argparse
import datetime
import numpy as np
from code.mesh import create_solid_mesh
from code.decompose import decompose
from code.graph import create_graph
from code.llm import run_llm
from code.load_data import load_confidence, load_mask
import trimesh
import pickle
import cv2

def create_output_directory(obj, output_idx, mode, no_object, model):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('outputs1', f'{obj}{output_idx}_{mode}_{current_time}')

    if model == "starling":
        output_dir += "_starling"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def run_pipeline(obj, task_string, data_dir, iter, output_idx, mode, threshold, no_object, model):
    variant = f"Running {obj} in mode {mode} task {output_idx} using {model}"
    if no_object:
        variant += " no_object"
    print(variant)
    obj_data_path = os.path.join(data_dir, obj)
    output_dir = create_output_directory(obj, output_idx, mode, no_object, model)
    binary_mask = load_mask(obj_data_path+'_mask')
    
    confidence_map = load_confidence(obj_data_path+'_confidence')
    if confidence_map is not None:
        points_value_255 = np.sum((confidence_map > 200) & (binary_mask == 1))
        total_mask_area = np.sum(binary_mask == 1)
        uncertainty = points_value_255 / total_mask_area if total_mask_area > 0 else 0
        print(f"uncertainty: {np.round(uncertainty, 2)}")
        if mode == '3d' and uncertainty > 0.15:
            print("use 2d")
            mode = '2d'
            
    graph_file = f'outputs_final/{obj}{iter}_{mode}_object/{obj}_graph.txt'
    if os.path.isfile(graph_file):
        print(f"load {obj} graph")
        with open(graph_file, 'r') as f:
            graph = f.read()
        shapes = cv2.imread(f'outputs_final/{obj}{iter}_{mode}_object/{obj}_shapes.png')
    else:
        pkl_file = f'outputs/{obj}{iter}_{mode}_object/{obj}_2d_hulls.pkl'
        if os.path.isfile(pkl_file):
            # print("already decomposed")
            with open(pkl_file, 'rb') as f:
                hulls = pickle.load(f)
        else:
            mesh_file = os.path.join(output_dir, f'{obj}_solid_mesh.obj')
            mesh = create_solid_mesh(obj_data_path, mesh_file, mode)
            # mesh = trimesh.load(f'outputs/{obj}{iter}_{mode}_object/{obj}_solid_mesh.obj', force="mesh")
            if not threshold:
                threshold = 0.2 if mode == '3d' else 0.15
            hulls = decompose(mesh, output_dir, mode, data_dir, threshold)
    
            if mode == '3d' and len(hulls) > 8:
                print("too many hulls, switching to 2d")
                # mode = '2d'
                # threshold = 0.15
                # hulls = decompose(mesh, output_dir, mode, data_dir, threshold)
            num_hulls = 0
            for hull in hulls:
                if hull.volume >= 500:
                    num_hulls += 1
            while num_hulls < 2 and threshold > 0.01:
                num_hulls = 0
                threshold = max(threshold - 0.025, 0.01)
                print(f"using threshold: {threshold}")
                hulls = decompose(mesh, output_dir, mode, data_dir, threshold)
                for hull in hulls:
                    if hull.volume >= 500:
                        num_hulls += 1
            print(f"num parts: {len(hulls)}")
        graph, shapes = create_graph(hulls, output_dir, obj_data_path, mode)
    grasp_point, img, most_likely_index = run_llm(graph, shapes, output_dir, obj_data_path, mode, no_object, task_string, model)
    return grasp_point, img, most_likely_index
    # TODO: grasp at this point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the 3D mesh pipeline')
    parser.add_argument('-o', '--obj', type=str, default='knife', help='Input object identifier')
    parser.add_argument('--task', type=str)
    parser.add_argument('--data_dir', type=str, default='data/', help='Directory where object data is located')
    parser.add_argument('-i', '--iter', type=int, help='iter')
    parser.add_argument('-m', '--mode', type=str, default='2d', help='3d or 2d (use depth or no depth)')
    parser.add_argument('-t', '--threshold', type=float, help='Threshold value')
    parser.add_argument('--no_object', action='store_true', help='Include to run without object')
    parser.add_argument('--model', type=str, default='gpt4', help='LLM inference model')
    args = parser.parse_args()
    run_pipeline(obj=args.obj, task_string=args.task, output_idx=1, data_dir=args.data_dir, iter=args.iter, mode=args.mode, threshold=args.threshold, no_object=args.no_object, model=args.model)
    