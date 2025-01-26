import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_training_metrics(log_file_path, title):
    timesteps = []
    D_data_loss = []
    D_total_loss = []
    G_discriminator_loss = []
    G_l2_loss_rel = []
    G_total_loss = []

    with open(log_file_path, 'r') as file:
        lines = file.readlines()

        timestep_pattern = re.compile(r't = (\d+) / \d+')
        D_data_loss_pattern = re.compile(r'\[D\] D_data_loss: ([\d\.]+)')
        D_total_loss_pattern = re.compile(r'\[D\] D_total_loss: ([\d\.]+)')
        G_discriminator_loss_pattern = re.compile(r'\[G\] G_discriminator_loss: ([\d\.]+)')
        G_l2_loss_rel_pattern = re.compile(r'\[G\] G_l2_loss_rel: ([\d\.]+)')
        G_total_loss_pattern = re.compile(r'\[G\] G_total_loss: ([\d\.]+)')
        model_save_pattern = re.compile(r'Saving checkpoint to (.+)')

        # Iterate over the lines and extract data
        for line in lines:
            timestep_match = timestep_pattern.search(line)
            if timestep_match:
                timesteps.append(int(timestep_match.group(1)))

            D_data_loss_match = D_data_loss_pattern.search(line)
            if D_data_loss_match:
                D_data_loss.append(float(D_data_loss_match.group(1)))

            D_total_loss_match = D_total_loss_pattern.search(line)
            if D_total_loss_match:
                D_total_loss.append(float(D_total_loss_match.group(1)))

            G_discriminator_loss_match = G_discriminator_loss_pattern.search(line)
            if G_discriminator_loss_match:
                G_discriminator_loss.append(float(G_discriminator_loss_match.group(1)))

            G_l2_loss_rel_match = G_l2_loss_rel_pattern.search(line)
            if G_l2_loss_rel_match:
                G_l2_loss_rel.append(float(G_l2_loss_rel_match.group(1)))

            G_total_loss_match = G_total_loss_pattern.search(line)
            if G_total_loss_match:
                G_total_loss.append(float(G_total_loss_match.group(1)))

    data = {
        'Timesteps': timesteps,
        'D_data_loss': D_data_loss,
        'D_total_loss': D_total_loss,
        'G_discriminator_loss': G_discriminator_loss,
        'G_l2_loss_rel': G_l2_loss_rel,
        'G_total_loss': G_total_loss
    }
    df = pd.DataFrame(data)

    plots_dir = os.path.join(os.path.dirname(log_file_path), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    sns.lineplot(x='Timesteps', y='D_data_loss', data=df)
    plt.title('D_data_loss')
    plt.savefig(os.path.join(plots_dir, f'{title}_D_data_loss.png'))

    plt.subplot(3, 2, 2)
    sns.lineplot(x='Timesteps', y='D_total_loss', data=df)
    plt.title('D_total_loss')
    plt.savefig(os.path.join(plots_dir, f'{title}_D_total_loss.png'))

    plt.subplot(3, 2, 3)
    sns.lineplot(x='Timesteps', y='G_discriminator_loss', data=df)
    plt.title('G_discriminator_loss')
    plt.savefig(os.path.join(plots_dir, f'{title}_G_discriminator_loss.png'))

    plt.subplot(3, 2, 4)
    sns.lineplot(x='Timesteps', y='G_l2_loss_rel', data=df)
    plt.title('G_l2_loss_rel')
    plt.savefig(os.path.join(plots_dir, f'{title}_G_l2_loss_rel.png'))

    plt.subplot(3, 2, 5)
    sns.lineplot(x='Timesteps', y='G_total_loss', data=df)
    plt.title('G_total_loss')
    plt.savefig(os.path.join(plots_dir, f'{title}_G_total_loss.png'))

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{title}_all_losses_combined.png'))
    plt.show()

    saved_models = []
    for line in lines:
        model_save_match = model_save_pattern.search(line)
        if model_save_match:
            saved_models.append(model_save_match.group(1))

    print("Last saved model steps:")
    for model in saved_models[-10:]:  # Adjust number as needed
        print(model)

if __name__ == '__main__':
    base_path = '/home/mattb/AnythingVisionaries/KI_GAN/scripts'  # Specify the base path
    plot_training_metrics(os.path.join(base_path, 'Model_Baseline_12_UniformNoise/log.txt'), '12 Step Prediction Baseline Model')
    plot_training_metrics(os.path.join(base_path, 'Model_Baseline_18_UniformNoise_Bs64/log.txt'), '18 Step Prediction Baseline Model')
    plot_training_metrics(os.path.join(base_path, 'Model_Baseline_24_UniformNoise_Bs64/log.txt'), '24 Step Prediction Baseline Model')
    plot_training_metrics(os.path.join(base_path, 'Model_NoTraff_12_UniformNoise_bs64/log.txt'), '12 Step Prediction No Traffic Encoder Model')
    plot_training_metrics(os.path.join(base_path, 'Model_NoTraff_18_UniformNoise_bs64/log.txt'), '18 Step Prediction No Traffic Encoder Model')
