# POMDP_Solver README

This document provides instructions on how to download and use the "POMDP_Solver" project. Follow these steps to get started:

## Step 1: Clone the Repository

To get started, you'll need to clone the project's GitHub repository. Use the following command to clone the repository, replacing `"git_url"` with the actual URL of the Git repository and `"branch_name"` with the desired branch name:

```shell
git clone "git_url" -b branch_name
```

## Step 2: Provide GitHub Token

When prompted for your username and password during the cloning process, you should use a GitHub token as the password. Make sure you have a GitHub token saved and use it instead of your password.

## Step 3: Set up the Development Environment

Before diving into the project, it is recommended to open the project in Visual Studio Code (VS Code). You can utilize the VS Code terminal for running commands and managing the project.

Inside the project directory, create a Python virtual environment (venv) using the following command:

```shell
python3 -m venv venv
```

## Step 4: Activate the Virtual Environment

Activate the virtual environment by running the following command in your terminal:

```shell
source ./venv/bin/activate
```

## Step 5: Install Project Requirements

With the virtual environment activated, you can now install the project's dependencies listed in the `requirements.txt` file:

```shell
pip install -r requirements.txt
```

## Step 6: Run Training

Now that you have set up the environment and installed the required packages, you can start running the training process. Use the following command to run the training script:

```shell
python3 ./main.py
```
## Step 7: Customize Training with Command Line Arguments

You can customize the training process using command line arguments when running the `main.py` script. Here are the available command line arguments along with their default values:

- `--gamma` (float): Discount factor (default: 0.95).
- `--lr` (float): Learning rate (default: 4e-4).
- `--batch_size` (int): Batch size (default: 64).
- `--time_steps` (int): Number of time steps (default: 4).
- `--eps` (float): Initial epsilon value (default: 1.0).
- `--eps_decay` (float): Epsilon decay rate (default: 0.9995).
- `--eps_min` (float): Minimum epsilon value (default: 0.01).
- `--model` (str): Model type (default: 'DTQN').
- `--env` (str): Environment type (default: 'Tiger').
- `--max_episodes` (int): Maximum number of episodes (default: 1500).

You can specify these arguments when running the training script. For example, to set a different learning rate, use:

```shell
python3 ./main.py --lr 0.001
```

During training, you'll see progress and updates on the screen. The resulting plot will be saved in the "results" directory within the project.

You are now ready to use the "POMDP_Solver" project. If you encounter any issues or have questions, please refer to the project's documentation or seek assistance from the project's maintainers. Enjoy working on your project!
