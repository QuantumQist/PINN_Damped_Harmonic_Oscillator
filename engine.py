import os
import shutil
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import utils

def train_model(model, training_params, system_params):
    num_epochs = training_params["num_epochs"]
    batch_size = training_params["batch_size"]
    lambda_1 = training_params["lambda_1"]
    lambda_2 = training_params["lambda_2"]
    lambda_3 = training_params["lambda_3"]
    training_t_max = training_params["t_max"] # Maximum time considered in evolution during training

    omega_0 = system_params["omega_0"]
    gamma = system_params["gamma"]

    t_test = torch.linspace(0, training_t_max, 1000).view(-1,1).to(model.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01)

    physics_loss_list, boundary_loss_list1, boundary_loss_list2 = [], [], []
    total_loss_list = []

    os.makedirs("plots", exist_ok=True)

    # Remove all files from "plots" directory
    for filename in os.listdir("plots"):
        file_path = os.path.join("plots", filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    torch.save(model.state_dict(), "best_model.pth")
    model.train()

    for epoch in tqdm(range(num_epochs)):
        model.train()
        # Forward pass - boundary loss
        boundary_t = torch.tensor([0.0], device=model.device, requires_grad=True).view(-1,1)
        output = model(boundary_t)
        boundary_loss_1 = (output - torch.ones_like(output, device=model.device))**2

        # Check velocity at initial point
        dx_dt = torch.autograd.grad(output, boundary_t, grad_outputs=torch.ones_like(output), create_graph=True)[0]
        boundary_loss_2 = (dx_dt - torch.zeros_like(dx_dt, device=model.device))**2

        boundary_loss_list1.append(lambda_2*boundary_loss_1.item())
        boundary_loss_list2.append(lambda_3*boundary_loss_2.item())

        # Forward pass - time samples
        t_tensor = torch.rand(batch_size, requires_grad=True, device=model.device).view(-1,1) * training_t_max
        output = model(t_tensor)
        # Derivatives
        dx_dt = torch.autograd.grad(output, t_tensor, grad_outputs=torch.ones_like(output), create_graph=True)[0]
        d2x_dt2 = torch.autograd.grad(dx_dt, t_tensor, grad_outputs=torch.ones_like(dx_dt), create_graph=True)[0]
        # Differential equation
        diff_eqn = d2x_dt2 + 2 * gamma * dx_dt + omega_0**2 * output
        # Loss
        physics_loss = torch.mean(diff_eqn**2)
        physics_loss_list.append(lambda_1*physics_loss.item())

        # Total loss
        total_loss = lambda_1 * physics_loss + lambda_2 * boundary_loss_1 + lambda_3 * boundary_loss_2
        total_loss_list.append(total_loss.item())

        # Update the model
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Make plot every 20 epochs and save it to "plots" folder
        if (epoch+1) % 20 == 0:
            model.eval()
            with torch.inference_mode():
                model_result = model(t_test).squeeze().cpu().numpy()
            plt.plot(t_test.squeeze().cpu().numpy(), model_result, label = "Model result")
            plt.plot(t_test.squeeze().cpu().numpy(), utils.ground_truth(t_test.squeeze().cpu().numpy(), omega_0, gamma), label = "Ground truth")
            plt.legend()
            plt.title(f"Epoch {epoch+1}")
            plt.xlabel("t")
            plt.ylabel("x(t)")
            plt.xlim(0, training_t_max)
            plt.ylim(-1.1, 1.1)
            file = f"plots/epoch_{epoch+1}.png"
            plt.savefig(f"plots/epoch_{epoch+1}.png")
            plt.close()

        if total_loss < model.best_loss:
            model.best_loss = total_loss
            torch.save(model.state_dict(), "best_model.pth")

    model.load_state_dict(torch.load("best_model.pth"))

    model.eval()
    fig, axs = plt.subplots(2, 1, figsize = (6, 6), sharex=True)
    axs[0].plot(physics_loss_list, label = "Physics loss")
    axs[0].plot(boundary_loss_list1, label = "Boundary loss - x(0)")
    axs[0].plot(boundary_loss_list2, label = "Boundary loss - dx/dt(0)")
    axs[0].legend()
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Loss")
    axs[1].plot(total_loss_list, label = "Total loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    axs[1].set_yscale("log")
    plt.show()