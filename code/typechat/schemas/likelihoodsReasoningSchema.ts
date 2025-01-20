// The following is a schema definition for reasoning about the likelihoods that each node/part is best to grasp for the task.

export interface LikelihoodsReasoning {
    reasoning: string; // Put your entire response to the prompt here, as a single natural language string reasoning about which node/part is best for the robot to grasp for the task
}
