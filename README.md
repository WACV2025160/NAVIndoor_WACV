# NAVIndoor

NavIndoor is a program developed to implement the NavImpaired method. It facilitates the acquisition of navigational data within procedurally generated environments and was created using Unity and its MLAgents library. The process begins with closed cells and a virtual agent equipped with a head-mounted camera (a), utilizes the Depth First Search algorithm for maze procedural generation (b), and populates it with obstacles and coins (c). Observations consist of the current scene view captured by a basic camera (d), and by a camera specifically designed to produce semantic segmentation maps, highlighting floor, walls, obstacles, and coins (e). Both views have dimensions of 128 x 128 x 3. The possible actions for the agent include moving forward, moving backward, rotating right, and rotating left.

<div align="center">
  <img src="https://github.com/PaperID1776/NAVIndoor/blob/main/images/maze_gen.png" alt="Navigation in the maze after training">
</div>

List of customizable parameters for the environments is presented below.


| Parameter            | Description                                                              | Range            | Default |
|----------------------|--------------------------------------------------------------------------|------------------|---------|
| **coin_proba**       | Probability of a coin appearing in each maze cell.                       | [0,1]          | 1       |
| **obstacle_proba**   | Probability of an obstacle appearing in each maze cell.                  | [0,1]           | 0.3     |
| **move_speed**       | The speed at which the agent moves forward or backward.                  | ≥0 | 1       |
| **turn_speed**       | The speed at which the agent rotates.                                    | ≥0               | 150     |
| **momentum**         | Inertial moment kept by the agent when changing direction.               | [0,1]           | 0       |
| **decreasing_reward**| Decreases the reward when agent stays in contact with a collider (wall/obstacle). | True/False       | False   |
| **coin_visible**     | Coins visibility by the camera.                                         | True/False       | True    |


The default reward function `R` is defined as follows:
- `-1` if the agent collides with an obstacle,
- `-0.5` if the agent collides with a wall,
- `5` if the agent collects a coin,
- `0` otherwise.


## Training in NavIndoor

We propose a `Demo_01` notebook implementing D3QN for training in NavIndoor.

<div align="center">
  <img src="https://github.com/PaperID1776/NAVIndoor/blob/main/images/arch2.png" alt="Navigation in the maze after training">
</div>


## Navigation in the maze after training

A model checkpoint is available in `checkpoint/checkpoint.pt` and allows to clip virtual navigation easily through `Demo_02` notebook.

<div align="center">
  <img src="https://github.com/PaperID1776/NAVIndoor/blob/main/images/explore.gif" alt="Navigation in the maze after training">
</div>

## Real world deployment

We demonstrated in our submission that the model's outputs correlate well with real-world characteristics for static images with a small field of view (FOV). A video was captured with a smartphone to showcase potential navigational features that could be extracted in real-time using NavImpaired through the 'Demo_03' notebook.

The generated figures include a dynamic plot of $V_{\theta}$ over time on the left, providing insights into the temporal evolution of the environment. In the center, the resized video is displayed.

On the right, the final image shows the input semantic segmentation maps used by NavImpaired. The length of the top arrow is proportional to $V_{\theta}(s)$, with its color indicating a simple thresholding mechanism applied to simulate a potential signal for sensory substitution systems.

Furthermore, the lengths of the bottom arrows represent the softmax values on $A_{\theta}(s,a)$ for the forward, backward, rotate left, and rotate right actions. The colored arrow, depicted in dark green, signifies the optimal action policy determined by the model. We observe a correlation between $A_{\theta}(s,forward)$, $V_{\theta}(s)$, and the path clearance in front of the cameraman.

<div align="center">
  <img src="https://github.com/PaperID1776/NAVIndoor/blob/main/video_processing/output_sample.gif" alt="Real world deployment">
</div>
