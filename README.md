# Dynamic Table Generation: Harnessing the Diffusion Model

## Running Sequence

1. **Prepare Dataset Labels**  
   Run the following script to prepare dataset labels:
python Dynamic_Table_Generation/utils/dataset_label_preparation.py
*Ensure the dataset is located at the path:* `./Dataset/PubTabNet/pubtabnet/pubtabnet`.


2. **Train the Model**  
To train the model, execute:
python Dynamic_Table_Generation/main.py


3. **Generate SSIM Score**  
To calculate the SSIM score for model evaluation, run:
python SSIM_Score/main.py


4. **Generate Table as Web Service**  
To generate tables as a web service, run:
python Table_Generation/app.py

Open the webpage and provide the necessary input to get generated tables.

---

### Label Preperation sample, training and validation loss curves


**Label Preparation**
<p align="center">
  <img src="Images/Labels_before_processing.png" width="400" alt="Labels before processing">
  <br>
  <em>Labels before processing</em>
</p>
<p align="center">
  <img src="Images/Labels_after_processing.png" width="400" alt="Labels after processing">
  <br>
  <em>Labels after processing</em>
</p>


**Training and Validation Loss**
<p align="center">
  <img src="Images/timesteps_10.png" width="400" alt="Timesteps - 10">
  <br>
  <em>Loss Curve for Timesteps - 10</em>
</p>
<p align="center">
  <img src="Images/timesteps_15.png" width="400" alt="Timesteps - 15">
  <br>
  <em>Loss Curve for Timesteps - 15</em>
</p>
<p align="center">
  <img src="Images/timesteps_20.png" width="400" alt="Timesteps - 20">
  <br>
  <em>Loss Curve for Timesteps - 20</em>
</p>
<p align="center">
  <img src="Images/timesteps_30.png" width="400" alt="Timesteps - 30">
  <br>
  <em>Loss Curve for Timesteps - 30</em>
</p>
<p align="center">
  <img src="Images/timesteps_40.png" width="400" alt="Timesteps - 40">
  <br>
  <em>Loss Curve for Timesteps - 40</em>
</p>
<p align="center">
  <img src="Images/timesteps_50.png" width="400" alt="Timesteps - 50">
  <br>
  <em>Loss Curve for Timesteps - 50</em>
</p>

### Sample Output

Below are samples of generated images via the web service:

**Generated images through web service**

<p align="center">
<img src="Images/generated_images_timesteps-10.png" width="600" alt="Generated Images for Timesteps - 10">
<br>
<em>Generated Images for Timesteps - 10</em>
</p>
<p align="center">
<img src="Images/generated_images_timesteps-15.png" width="600" alt="Generated Images for Timesteps - 15">
<br>
<em>Generated Images for Timesteps - 15</em>
</p>
<p align="center">
<img src="Images/generated_images_timesteps-20.png" width="600" alt="Generated Images for Timesteps - 20">
<br>
<em>Generated Images for Timesteps - 20</em>
</p>
<p align="center">
<img src="Images/generated_images_timesteps-30.png" width="600" alt="Generated Images for Timesteps - 30">
<br>
<em>Generated Images for Timesteps - 30</em>
</p>
<p align="center">
<img src="Images/generated_images_timesteps-40.png" width="600" alt="Generated Images for Timesteps - 40">
<br>
<em>Generated Images for Timesteps - 40</em>
</p>
<p align="center">
<img src="Images/generated_images_timesteps-50.png" width="600" alt="Generated Images for Timesteps - 50">
<br>
<em>Generated Images for Timesteps - 50</em>
</p>


