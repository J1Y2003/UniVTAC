# AI Agent (Claude) Implementation Guideline: GR00T-N1.7 on UniVTAC

## 1. Primary Objective
Your goal is to write a Python-based evaluation pipeline to benchmark the `nvidia/GR00T-N1.7-3B` foundation model on the `UniVTAC` benchmark. 

Specifically, you need to write the codebase to conduct an ablation study comparing two setups:
*   **Baseline (No Tactile):** Standard GR00T inputs (Vision, Text Instruction, 1D Proprioception State, Embodiment ID).
*   **Ablation (Tactile-Enabled):** The 1D Proprioception State vector must be concatenated with the flattened tactile sensor arrays provided natively by the UniVTAC simulation environments.

## 2. Server & Execution Constraints
*   The code will be executed on a highly restricted Kakao SLURM cluster.
*   Do not write interactive scripts. All evaluations must be designed to run headlessly via SLURM batch submission (`sbatch`).
*   Ensure proper memory management and avoid heavy I/O operations, as the login nodes have strict memory limits.

## 3. Implementation Requirements
Please generate the necessary Python scripts and Gym/Gymnasium wrappers to accomplish the following:

1.  **Environment Wrapper:** Create a custom Gym wrapper for UniVTAC that standardizes the observation space. It must include a flag to toggle the tactile modality (flattening and concatenating the UniVTAC tactile array to the state vector).
2.  **Action Handling:** GR00T utilizes a Flow Matching Action Transformer (DiT) that outputs action chunks (e.g., 16-step horizons). Implement a receding horizon control loop to process these chunks and step the environment appropriately.
3.  **Evaluation Loop:** Write a robust evaluation script that seeds the environment, runs a specified number of episodes, and logs success rates and rewards.

## 4. Suggested Sources and Repositories
To build this, please leverage existing open-source frameworks rather than writing everything from scratch. You can reference or integrate with the following:
*   **Hugging Face `lerobot`:** For dataset handling, model instantiation, and standard VLA evaluation loops.
*   **NVIDIA `gr00t-leapp-export` (gr00t_workflow_0.1):** Specifically, look into how they handle modality configurations (e.g., `new_embodiment_config_defaults.py`) and custom `NEW_EMBODIMENT` tags for passing variable-length 1D state vectors to the model.
*   **UniVTAC Official Repository:** For the base environment API and observation space structure.

**Important Disclaimer:** The resources and methodologies listed in this document are based on preliminary architectural research. They are **not conclusive**, and there may be more up-to-date, efficient, or better-suited open-source repositories and wrappers available for this specific task. Please search your knowledge base and utilize the best available methods to achieve the stated objective.