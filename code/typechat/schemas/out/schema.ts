// The following is a schema definition for assigning a single likelihood (value) to each node (key).

export interface NodeLikelihoods {
    rect0: number; // put here only the likelihood between 0 and 1 that the robot should grasp rect0 and nothing else.
    rect1: number; // put here only the likelihood between 0 and 1 that the robot should grasp rect1 and nothing else.
}