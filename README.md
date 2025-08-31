# Human Pose Classifier using Visual Transformers (ViT)

This project implements a **human pose classifier** using Visual Transformers (ViT), covering the full pipeline from data preprocessing to real-world inference.

The best-performing model achieves high classification metrics:

<div align="center">

| Metric | Value |
| --------- | ------ |
| Accuracy | 0.791 |
| Precision (weighted) | 0.792 |
| Recall (weighted) | 0.791 |
| F1-Score (weighted) | 0.790 |

</div>

The model can be deployed on an **Amazon EC2 Instance**, and a live prototype is accessible via **Streamlit Server**: https://cv-human-pose-classifier-vit.streamlit.app/

<div align="center">

![Demo](assets/demo.gif)

</div>

## Key Features & Technologies
- **Dataset**: Uses the [Bingsu/Human_Action_Recognition](https://huggingface.co/datasets/Bingsu/Human_Action_Recognition) dataset loaded and split via **Hugging Face** library.
- **Model**: **Visual Transformers (ViT)** for classifying images into 15 human action classes: *"calling", "clapping", "cycling", "dancing", "drinking", "eating", "fighting", "hugging", "laughing", "listening_to_music", "running", "sitting", "sleeping", "texting" and "using_laptop"*.
- **AWS Integration**: Trained models are uploaded to **S3 Bucket** using **boto3**.
- **Deployment**: Model can be served on **EC2 Instance**
- **Web Interface**: Interactive inference via **FastAPI** backend and **Streamlit** frontend.

## AWS Services Configuration for Model Deployment
1. Create a User using **AWS IAM Service** with the following permissions:
    - `AmazonEC2FullAccess`
    - `AmazonS3FullAccess`

    *(You can also create a custom policy for more restricted access if needed.)*

2. **Generate an Access Key** for this IAM user and **save the following securely**:
    - Access Key ID
    - Secret Access Key

3. The S3 Bucket will be created dynamically during the execution of the [testing_pipeline.py](src/testing_pipeline.py). If the best found model during training achieves the required score in the [testing_config.json](config/testing_config.json), it will be uploaded to this bucket.\
⚠️ (*Make sure the bucket name you configure is globally unique to avoid conflicts.*)

4. Create an EC2 instance with the following specifications:
    - **AMI**: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)
    - **Instance type**: `t3.medium`
    - **Key Pair**: Use the key pair you downloaded to connect via SSH
    - **Security Group**:
        - Allow inbound rules for ports: `22`, `80`, `8501`, `8502` (all TCP)
    - **Storage**: 120 GiB (gp3)

5. Run the EC2 instance and interact with it via SSH or any other remote access method to set up the environment:
    - Create a working directory:
        ```bash
        mkdir mlops
        cd mlops
        ```
    - Clone your GitHub repo:
        ```bash
        git clone https://github.com/Lahdhirim/CV-human-pose-classifier-ViT-aws.git
        cd CV-human-pose-classifier-ViT-aws
        ```
    - Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    - Configure the AWS credentials using AWS CLI:
        ```bash
        aws configure Press ENTER
        AWS Access Key ID: ************
        AWS Secret Access Key: ************
        Default region name: Press ENTER
        Default output format: Press ENTER
        ```
    - Add Streamlit to PATH (for command-line use):
        ```bash
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
        ```

6. (Optional but recommended) Run the application from the terminal to ensure that everything is working correctly:
    ```bash
    python3 src/web_app/server.py
    ```
    in a new terminal:
    ```bash
    streamlit run src/web_app/interface.py
    ```
    Normally, at this stage, if everything works fine, the application is accessible at `http://<public IPv4 address>:8501`

7. Automatically launch Streamlit on instance reboot:
    - Create the startup script:
      ```bash
      nano /home/ubuntu/start_streamlit.sh
      ```

      Paste the following:
      ```bash
        #!/bin/bash
        cd /home/ubuntu/mlops/CV-human-pose-classifier-ViT-aws
        source /home/ubuntu/.bashrc
        nohup /usr/bin/python3 src/web_app/server.py >> /home/ubuntu/fastapi.log 2>&1 &
        sleep 20
        nohup /home/ubuntu/.local/bin/streamlit run src/web_app/interface.py --server.port 8501 >> /home/ubuntu/streamlit.log 2>&1 &
      ```

    - Make the script executable:
        ```bash
        chmod +x /home/ubuntu/start_streamlit.sh
        ```
    - Add the script to `crontab` for reboot:
        ```bash
        crontab -e
        ```
        Add the following line at the end of the file:
        ```bash
        @reboot /home/ubuntu/start_streamlit.sh
        ```

Each time the instance is rebooted, Streamlit will automatically launch the web application at the address  `http://<public IPv4 address>:8501`. Two log files named  `fastapi.log` and `streamlit.log` will be created in the  `/home/ubuntu` directory. These files can be used to monitor the application’s status and debug any errors.\
The application will be publicly accessible to anyone with the instance’s public IP address. Access can be controlled via the EC2 Security Group:
- To allow access from any IP address, set the Source  `to 0.0.0.0/0` on TCP port  `8501`.\
    ⚠️ Use  `0.0.0.0/0` only if you're aware of the security implications. For more restricted access, specify your own IP or a limited range.

## Human Pose Classification & Beyond

This classifier is designed not only for human action recognition but also as a **flexible image classification framework**. Thanks to its modular pipeline and configuration-driven design:

- **Configurable Pipelines**: Each stage of the workflow (preprocessing, training, testing and inference) is controlled by its own configuration file (`preprocessing_config.json`, `training_config.json`, `testing_config.json`, `inference_config.json`).
- **No Code Changes Required**: You can adapt the application to new datasets or classification tasks simply by updating the configuration files.
- **End-to-End Workflow**: From data preprocessing to model training, evaluation, and deployment, all steps are fully automated and modular.
- **Rapid Deployment**: The same FastAPI + Streamlit interface can serve any trained model without modification, making it suitable for a wide range of computer vision tasks beyond human pose classification.
