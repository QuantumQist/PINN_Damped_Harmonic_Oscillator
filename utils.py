import moviepy.video.io.ImageSequenceClip
import re
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def natural_sort_key(s):
    """
    This function is used to sort the plots in correct order
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def make_video(images_dir, video_name, fps = 15):
    """
    Makes a video from a list of images.

    Parameters:
        images_dir: Path to the folder containing the images.
        video_name: Name of the file with saved video.
        fps: Number of frames per second.
    """
    image_files = [os.path.join(images_dir, img) for img in sorted(os.listdir(images_dir), key=natural_sort_key) if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(video_name)

def ground_truth(t, omega_0, gamma):
    """
    Returns the ground truth trajectory of underdamped harmonic oscillator.

    Parameters:
        t: Time. Can be either single parameter or an array.
        omega_0: Angular frequency of the oscillator without damping.
        gamma: Damping parameter.

    Returns:
        Ground truth trajectory. (float or array)
    """
    assert omega_0 > gamma, "omega_0 must be greater than gamma (weak damping regime)"

    omega = np.sqrt(omega_0**2 - gamma**2)
    exponential_term = np.exp(-gamma * t)
    phase = np.arctan(- gamma / omega)
    return exponential_term * np.cos(omega * t + phase)

def plot_model_result(model, t_list, omega_0, gamma):
    """
    Plots the result produced by the PyTorch model and the ground truth trajectory.

    Parameters:
        model: PyTorch model.
        t_list: List of times.
        omega_0: Angular frequency of the oscillator.
        gamma: Damping parameter.
    """
    model.eval()
    t_tensor = torch.tensor(t_list, dtype = torch.float).view(-1,1).to(model.device)
    with torch.inference_mode():
        model_result = model(t_tensor).squeeze().cpu().numpy()

    plt.plot(t_list, model_result, label = "Model result")
    plt.plot(t_list, ground_truth(t_list, omega_0, gamma), label = "Ground truth")
    plt.legend()