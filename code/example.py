from typechat.typechat import TypeChat
from keys import API_KEY, ORG_KEY
import re
import ast 

def parse_nodes(graph_text):
        nodes_str = re.search(r"Nodes of the graph:\s*(\[.*\])", graph_text).group(1)
        nodes = ast.literal_eval(nodes_str)
        node_names = [node[0] for node in nodes]

        return node_names

def generate_ts_schema_for_likelihoods(nodes):
    # Start the interface definition
    schema = "// The following is a schema definition for assigning a single (unique) likelihood to each node in the graph.\n\n"
    schema += "export interface NodeLikelihoods {\n"
    # For each node, add a property to the schema
    for node in nodes:
        schema += f"    {node}: number; // put only one number here, the likelihood between 0 and 1 that the robot should grasp this node\n"
    # Close the interface
    schema += "}"
    return 'NodeLikelihoods', schema

def generate_ts_schema_for_parts(nodes):
    # Start the interface definition
    schema = "// The following is a schema definition for assigning a single semantic part to each node in the graph.\n\n"
    schema = "export interface NodePartsAssignment {\n"
    # For each node, add a property to the schema
    for node in nodes:
        schema += f"    {node}: string;  // put here the single semantic part assigned to this node as a string\n" 
    # Close the interface
    schema += "}"
    return 'NodePartsAssignment', schema

if __name__ == "__main__":
    # First, create a TypeChat instance
    ts = TypeChat()
    # Set up the language model that you want to use
    # Note: If you are utilizing local LLMs (e.g. through FastChat), you can set base_url to the URL of your local LLM
    # Note: You can enable json_mode if you want, however, only gpt-4-1106-preview supports this
    # ts.createLanguageModel(model="gpt-4-0125-preview", api_key=API_KEY, org_key=ORG_KEY, use_json_mode=True)
    ts.createLanguageModel(model="Starling-LM-7B-alpha", api_key=API_KEY, org_key=ORG_KEY, base_url="http://localhost:23002/v1")
    obj = "hammer"
    with open("/home/samli/symbolic_llm_grasp/outputs/hammer1_3d_object/hammer_graph.txt", "r") as file:
        graph = file.read()
    nodes = parse_nodes(graph)

    # parts reasoning
    parts_reasoning_schema = "./typechat/schemas/partsReasoningSchema.ts"
    ts.loadSchema(parts_reasoning_schema)
    parts_reasoning_name = "PartsReasoning"
    tns = ts.createJsonTranslator(name=parts_reasoning_name)
    
    request = [
        {"role": "user", "content": f"Here is an object '{obj}' decomposed into convex parts, each represented as a node in a graph with attributes (shape, aspect ratio, color, area). The aspect ratio attribute is defined by long axis over short axis. Edges in the graph represent parts that are connected within the object.\n\n"},
    ]
    request[0]["content"] += graph 
    request[0]["content"] += "\n\nReason about what part each node may represent in the object. Be succinct, give only a short one sentence explanation for each node." 
    return_query = False
    response = tns.translate(request, image=None, return_query=return_query, free_form=True)
    if return_query:
        print(response)
    else:
        if response.success:
            parts_reasoning_response = response.data["reasoning"]
            print("Parts reasoning response:")
            print(parts_reasoning_response)
        else:
            print("parts reasoning error:", response.message)

    # now assign semantic parts to each node
    parts_name, ts_schema_parts = generate_ts_schema_for_parts(nodes)
    ts.loadSchema(schema=ts_schema_parts)
    tns = ts.createJsonTranslator(name=parts_name)
    assign_part_request = request + [
            {"role": "assistant", "content": parts_reasoning_response},
            {"role": "user", "content": f"Assign a semantic part to each node."},
        ]
    return_query = False
    response = tns.translate(assign_part_request, image=None, return_query=return_query, free_form=True)

    if return_query:
        print(response)
    else:
        if response.success:
            print("\nPart assignment response:")
            print(response.data)
        else:
            print("part assignment error:", response.message)

    likelihoods_reasoning_schema = "./typechat/schemas/likelihoodsReasoningSchema.ts"
    likelihoods_reasoning_name = "LikelihoodsReasoning"
    ts.loadSchema(likelihoods_reasoning_schema)
    tns = ts.createJsonTranslator(name=likelihoods_reasoning_name)
    # task = f"Imagine you are a robot hand and want to lift the {obj} in a safe and secure manner. Reason about how likely each node/part is best for the robot to grasp. Be succinct, give only a short one sentence explanation for each node."
    # task = f"Imagine you are a robot hand and want to pick up the {obj} and hand it to a human so that the human can easily take and grip it by a proper part for usage, in a safe manner without being obstructed. Reason about how likely each node/part is best for the ROBOT to grasp. Be succinct, give only a short one sentence explanation for each node."
    task = f"Imagine you are a robot hand tasked with picking up a {obj} and handing it to a human in such a way that the human can easily and safely grip the proper part for usage, avoiding obstructions. Identify the best part of the {obj} for the robot to grasp and provide a short explanation for your choice, considering the ease of handover and safety."
    request += [
        {"role": "assistant", "content": parts_reasoning_response},
        {"role": "user", "content": task},
    ]
    return_query = False
    response = tns.translate(request, image=None, return_query=return_query, free_form=True)
    if return_query:
        print(response)
    else:
        if response.success:
            # The response data is a dictionary with the keys being the names of the fields in your schema
            likelihoods_reasoning_response = response.data["reasoning"]
            print("\nLikelihoods reasoning response:")
            print(likelihoods_reasoning_response)
        else:
            print("Error:", response.message)

    # now assign numeric likelihoods to each part
    likelihoods_name, ts_schema_likelihoods = generate_ts_schema_for_likelihoods(nodes)
    ts.loadSchema(schema=ts_schema_likelihoods)
    tns = ts.createJsonTranslator(name=likelihoods_name)
    # task = f"\n\nImagine you are a robot hand and want to lift the {obj} in a safe and secure manner. Give (distinct) likelihood estimates between 0 and 1 that each part is the correct part for the robot to grasp."
    # task = f"\n\nImagine you are a robot hand and want to pick up the {obj} and hand it to a human, in a way such that the human can receive and grasp it by a safe and secure part. Give (distinct) likelihood estimates between 0 and 1 that each part is the correct part for the robot to grasp."
    task = f"\n\nAssign a likelihood to each node."
    request += [
        {"role": "assistant", "content": likelihoods_reasoning_response},
        {"role": "user", "content": task},
    ]
    return_query = False
    response = tns.translate(request, image=None, return_query=return_query, free_form=True)
    # The response object has a few fileds:
    # - success: True if the request was successful
    # - data: The response data
    # - error: The error message, if any
    if return_query:
        print(response)
    else:
        if response.success:
            # The response data is a dictionary with the keys being the names of the fields in your schema
            print("\nLikelihoods assignment response:")
            print(response.data)
        else:
            print("Error:", response.message)
    