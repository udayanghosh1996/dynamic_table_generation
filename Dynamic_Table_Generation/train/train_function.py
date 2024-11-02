import os
import torch
import torch.nn as nn
import torch.quantization as quant
from transformers import CLIPTextModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from tqdm import tqdm
import pandas as pd
import numpy as np

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

model_path = os.path.join(os.getcwd(), "models")
log_path = os.path.join(os.getcwd(), 'logs')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calibrate_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            true_image = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            text_embeddings = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            text_embeddings = projection_layer(text_embeddings).to(device)

            batch_size = true_image.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()

            noise = torch.randn_like(true_image)
            noisy_image = scheduler.add_noise(true_image, noise, timesteps)

            expanded_timesteps = expand_timesteps(timesteps, batch_size, true_image.shape[2], true_image.shape[3]).to(
                device)

            noisy_input = torch.cat([noisy_image, expanded_timesteps], dim=1)

            model(noisy_input, timesteps, encoder_hidden_states=text_embeddings)


def quantize_model(model, dataloader):
    model.eval()
    model.qconfig = quant.get_default_qconfig('fbgemm')
    model = quant.prepare(model, inplace=False)

    calibrate_model(model, dataloader)

    model = quant.convert(model, inplace=False)
    return model


def expand_timesteps(timesteps, batch_size, height, width):

    timesteps = timesteps.view(batch_size, 1, 1, 1).float()
    return timesteps.expand(batch_size, 1, height, width)


class ProjectTextEmbeddings(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectTextEmbeddings, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def train_function(train_loader, val_loader, epoches, lr, time_step):
    global text_encoder, projection_layer, scheduler

    model = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", addition_embed_type='text',
                                                 low_cpu_mem_usage=False).to(device)
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(time_step)


    projection_layer = ProjectTextEmbeddings(768, 2048).to(device)

    model = quantize_model(model, train_loader)



    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    num_train_epochs = epoches

    epoches_list = []
    train_loss_list = []
    val_losses_list = []

    for epoch in range(num_train_epochs):
        model.train()
        text_encoder.train()
        training_loss = 0.0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            true_image = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            text_embeddings = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            text_embeddings = projection_layer(text_embeddings).to(device)  # Project text embeddings and move to device

            batch_size = true_image.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()

            noise = torch.randn_like(true_image)
            noisy_image = scheduler.add_noise(true_image, noise, timesteps)

            expanded_timesteps = expand_timesteps(timesteps, batch_size, true_image.shape[2], true_image.shape[3]).to(
                device)

            noisy_input = torch.cat([noisy_image, expanded_timesteps], dim=1)

            pred_image = model(noisy_input, timesteps, encoder_hidden_states=text_embeddings).sample

            loss = criterion(pred_image, true_image)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        avg_train_loss = training_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_train_epochs} - Train Loss: {avg_train_loss}")
        train_loss_list.append(avg_train_loss)
        epoches_list.append(epoch)

        model.eval()
        with torch.no_grad():
            val_avg_loss = 0.0
            for val_batch in tqdm(val_loader):
                true_val_image = val_batch["pixel_values"].to(device)
                val_input_ids = val_batch["input_ids"].to(device)
                val_attention_mask = val_batch["attention_mask"].to(device)

                val_text_embeddings = text_encoder(input_ids=val_input_ids,
                                                   attention_mask=val_attention_mask).last_hidden_state
                val_text_embeddings = projection_layer(val_text_embeddings).to(
                    device)

                val_batch_size = true_val_image.shape[0]
                val_timesteps = torch.randint(0, scheduler.num_train_timesteps, (val_batch_size,), device=device).long()

                val_noise = torch.randn_like(true_val_image)
                val_noisy_image = scheduler.add_noise(true_val_image, val_noise, val_timesteps)

                val_expanded_timesteps = expand_timesteps(val_timesteps, val_batch_size, true_val_image.shape[2],
                                                          true_val_image.shape[3]).to(device)

                val_noisy_input = torch.cat([val_noisy_image, val_expanded_timesteps], dim=1)

                val_pred_image = model(val_noisy_input, val_timesteps, encoder_hidden_states=val_text_embeddings).sample

                val_loss = criterion(val_pred_image, true_val_image)
                val_avg_loss += val_loss.item()

            val_avg_loss /= len(val_loader)
            print(f"Epoch {epoch + 1}/{num_train_epochs} - Validation Loss: {val_avg_loss}")
            val_losses_list.append(val_avg_loss)

    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(os.path.join(model_path, 'Stable_Diffusion_Model'))
    text_encoder.save_pretrained(os.path.join(model_path, 'Stable_Diffusion_text_encoder'))
    scheduler.save_pretrained(os.path.join(model_path, 'Stable_Diffusion_scheduler'))

    logs_arr = np.array([epoches_list, train_loss_list, val_losses_list]).T
    logs = pd.DataFrame(logs_arr, columns=['Epoches', 'Training_Loss', 'Validation_Loss'])
    os.makedirs(os.path.join(log_path, 'Timesteps_'+str(time_step)), exist_ok=True)
    logs.to_csv(os.path.join(os.path.join(log_path, 'Timesteps_'+str(time_step)), 'Loss_curve_parameters.csv'))
