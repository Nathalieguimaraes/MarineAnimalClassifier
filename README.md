# Marine_Animal_Classifier
This project is a service for marine animals, classification, using a DL model fine-tuned.

<h1>Marine Animal Classifier</h1>

This project is a web-based classifier for marine animals, utilizing a deep learning model fine-tuned on a custom dataset sourced from Kaggle. It allows users to upload an image and receive predictions of the type of marine animal in the image. The application is packaged as a Docker container and deployed to a local Kubernetes cluster using Minikube.

Project Overview
Model Fine-Tuning: A pre-trained ResNet50 model was fine-tuned on a custom dataset of marine animals sourced from Kaggle.
Docker Containerization: The application was containerized using Docker.
Kubernetes Deployment: The containerized app was deployed to a local Kubernetes environment using Minikube.
Service Exposure: The application was exposed via a NodePort service to allow access from a web browser.

<h1>1. Prerequisites</h1>
Ensure you have the following installed:

Docker: For building and running containers.
Python 3.12: For running scripts and dependencies.
Minikube: For running a local Kubernetes cluster.
kubectl: For Kubernetes commands.
Kaggle API Key: To download the dataset from Kaggle.

<h1>2. Model Fine-Tuning</h1>
The model was fine-tuned using a marine animal dataset from Kaggle. You can find the dataset at this Kaggle link.

Steps to Fine-Tune the Model
Download Dataset from Kaggle: https://www.kaggle.com/discussions/general/400009

Use your Kaggle API key to download the dataset.
Place the dataset in the directory app/Marine_animals_dataset.
Run train.py script

A train.py script was created to load and fine-tune a ResNet50 model on this dataset.
The dataset was split into training, validation, and test sets.
The training was conducted for 50 epochs, and the final model was saved in the models directory.
Save the Fine-Tuned Model.
Ensure that the trained model is saved in app/models/fine_tuned_marine_animal_classifier.pth.

<h1>3. Application Setup</h1>
Directory Structure
Your directory structure should look like this:

project/
│<br>
├── app/<br>
│   ├── Dockerfile<br>
│   ├── main.py<br>
│   ├── model.py<br>
│   ├── train.py<br>
│   ├── tests/<br>
│   │   └── test_classify.py<br>
│   ├── models/<br>
│   │   └── fine_tuned_marine_animal_classifier.pth<br>
│   ├── Marine_animals_dataset/<br>
│   └── requirements.txt<br>

Dockerfile
The Dockerfile in the app directory is used to build the Docker container for the application. Ensure that main.py, model.py, and all necessary dependencies in requirements.txt are set up correctly.

To build the Docker image, use:
docker build -t marine_animal_classifier .

<h1>4. Running the Application Locally with Docker</h1>
Build and Run Docker Container:
docker build -t marine_animal_classifier .
docker run -p 8000:8000 -v "C:/Users/nsguimaraes/Documents/project/app/Marine_animals_dataset:/data/Marine_animals_dataset" marine_animal_classifier

Access the Application:
The application is accessible at http://localhost:8000/docs via Swagger UI.

<h1>5. Kubernetes Deployment with Minikube</h1>
Start Minikube:
minikube start --memory=4096 --cpus=2
Load Docker Image into Minikube:
minikube image load marine_animal_classifier
Apply Kubernetes Deployment:
kubectl apply -f deployment.yaml
Check Deployment and Pod Status:
kubectl get deployments
kubectl get pods
Expose the Service:
kubectl expose deployment marine-animal-classifier --type=NodePort --port=8000
Access the Application:
minikube service marine-animal-classifier
This command will open the application in your browser or display the accessible URL.

<h1>6. Testing the API</h1>
Running Tests
Run Unit Tests:

Ensure tests are located in the tests directory and named appropriately (e.g., test_classify.py).
pytest tests/
Sample Test:
A sample test was created to verify the classification endpoint. Ensure test_classify.py points to a valid image for testing.

<h1>7. Bonus: Pre-commit Hooks and Continuous Integration</h1>
To maintain code quality, pre-commit hooks were added using pre-commit and mypy for static type checking.

Install Pre-commit:
pre-commit install
Run Hooks Manually:
pre-commit run --all-files









