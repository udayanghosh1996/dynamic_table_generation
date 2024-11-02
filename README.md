# Dynamic Table Generation: Harnessing the Diffusion Model



## Running Sequence

1. **Prepare Dataset Labels**  
   Run the following script to prepare dataset labels:
python Dynamic_Table_Generation/utils/dataset_label_preparation.py

*Ensure the dataset is located at the path:* `./Dataset/PubTabNet/pubtabnet/pubtabnet`.

*Label Preperation*
![Labels before processing](Images/Labels_before_processing.png)
![Labels after processing](Images/Labels_after_processing.png)


2. **Train the Model**  
To train the model, execute:
python Dynamic_Table_Generation/main.py

*Training and Validation Loss*
![Timesteps - 10](Images/timesteps_10.png)
![Timesteps - 15](Images/timesteps_15.png)
![Timesteps - 20](Images/timesteps_20.png)
![Timesteps - 30](Images/timesteps_30.png)
![Timesteps - 40](Images/timesteps_40.png)
![Timesteps - 50](Images/timesteps_50.png)



3. **Generate SSIM Score**  
To calculate the SSIM score for model evaluation, run:
python SSIM_Score/main.py


4. **Generate Table as Web Service**  
To generate tables as a web service, run:
python Table_Generation/app.py
Open the webpage and provide the necessary input to get generated tables.

---

### Sample Output

Below are sample of generated images via the web service:

**Generated images through web service**

![Generated Images for Timesteps - 10](Images/generated_images_timesteps-10.png)
![Generated Images for Timesteps - 15](Images/generated_images_timesteps-15.png)
![Generated Images for Timesteps - 20](Images/generated_images_timesteps-20.png)
![Generated Images for Timesteps - 30](Images/generated_images_timesteps-30.png)
![Generated Images for Timesteps - 40](Images/generated_images_timesteps-40.png)
![Generated Images for Timesteps - 50](Images/generated_images_timesteps-50.png)
