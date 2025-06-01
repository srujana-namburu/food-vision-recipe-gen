# Food Vision Recipe Generator

## Project info

**URL**: https://lovable.dev/projects/6f1c8818-6565-4efb-9e77-f505aa3a3232

## How can I edit this code?

There are several ways of editing your application.

**Use Lovable**

Simply visit the [Lovable Project](https://lovable.dev/projects/6f1c8818-6565-4efb-9e77-f505aa3a3232) and start prompting.

Changes made via Lovable will be committed automatically to this repo.

**Use your preferred IDE**

If you want to work locally using your own IDE, you can clone this repo and push changes. Pushed changes will also be reflected in Lovable.

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Start the development server with auto-reloading and an instant preview.
npm run dev
```

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

## What technologies are used for this project?

This project is built with:

- Vite, TypeScript, React, shadcn-ui, and Tailwind CSS for the frontend
- Python Flask backend with the following AI models:
  - VisionEncoderDecoderModel for image captioning
  - DetrForObjectDetection for ingredient detection
  - Mistral-7B-Instruct for recipe generation

## How to run the application

### Prerequisites

- Node.js & npm for the frontend
- Python 3.8+ with pip for the backend
- Webcam for capturing food images

### Running the backend

```sh
# Navigate to the backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Start the Flask backend server
python app.py
# Or use the convenience script
./start_backend.sh
```

The backend server will run on http://localhost:5000

### Running the frontend

```sh
# In a new terminal window, install frontend dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at http://localhost:5173

### Using the application

1. Open the frontend URL in your browser
2. Allow camera access when prompted
3. Point your camera at food items
4. Click the capture button
5. The application will detect ingredients and generate a recipe

## Note on AI Models

The first time you run the backend, it will download the required AI models which may take some time depending on your internet connection. These models require significant disk space and memory to run.
