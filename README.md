# 👁️ OcuSense — Retinal Disease Detection

[![Python](https://img.shields.io/badge/python-3.10-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-v3.0-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Next.js](https://img.shields.io/badge/next.js-v14-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v2.0-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-v5.0-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-v3.0-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

OcuSense is a state-of-the-art retinal disease detection system that leverages deep learning to analyze retinal fundus images. Featuring a high-performance **Flask backend** powered by a custom-trained **EfficientNet-B3** architecture, and a modern, high-fidelity **Next.js 14 App Router** dashboard styled with Tailwind CSS.

> ⚠️ **DISCLAIMER:** This project is developed purely for research and educational purposes. It is **NOT** a certified medical device and should not be used as a substitute for professional clinical diagnosis or consultation.

---

## 🛠️ How It Works (System Architecture)

Here is a simplified high-level look at how image submissions traverse the application layers to produce medical classification insights:

```text
               +--------------------------+
               |  Retinal Fundus Image    |
               | (JPG, JPEG, or PNG scan) |
               +------------+-------------+
                            |
                            v [Drag & Drop Upload]
               +------------+-------------+
               |   Next.js Frontend UI    |
               | (Client Browser Interface)|
               +------------+-------------+
                            |
                            | POST /predict (FormData w/ 'image')
                            v
               +------------+-------------+
               |    Flask Backend API     |
               |  (Port 5000 CORS Router) |
               +------------+-------------+
                            |
                            v [Inference Transforms & PyTorch Model]
               +------------+-------------+
               |  Custom EfficientNet-B3  |
               | (Calculates Probabilities)|
               +------------+-------------+
                            |
                            | Returns JSON Prediction Package
                            v
               +------------+-------------+
               |    Next.js Result View   |
               | (Probabilities Chart/Badges)|
               +--------------------------+
```

---

## 🚀 Getting Started

Follow these step-by-step setup guides to spin up both the backend server and the frontend client on your local computer.

### 🐍 1. Backend Setup (Flask API)

The backend runs on Python (version 3.10 is recommended) and uses PyTorch to process the neural network inference computations.

1. **Open your terminal** and change directories into the `backend` folder:
   ```bash
   cd backend
   ```

2. **Create a fresh virtual environment** named `backendvenv` to isolate the project dependencies:
   ```bash
   python -m venv backendvenv
   ```

3. **Activate the virtual environment**:
   * **Windows (PowerShell):**
     ```powershell
     backendvenv\Scripts\Activate.ps1
     ```
   * **Mac / Linux:**
     ```bash
     source backendvenv/bin/activate
     ```
   * *Once activated, your terminal prompt will be prefixed with `(backendvenv)`.*

4. **Install all required libraries** using `pip`:
   ```bash
   pip install flask flask-cors torch torchvision pillow
   ```

5. **Launch the Flask application server**:
   ```bash
   python main.py
   ```
   * *The server will start up and listen for network requests on [http://localhost:5000](http://localhost:5000).*

---

### ⚛️ 2. Frontend Setup (Next.js Dashboard)

The frontend uses Node.js. Make sure you have Node.js installed on your computer before running these commands.

1. **Open a second terminal window** and change directories into the `frontend` folder:
   ```bash
   cd frontend
   ```

2. **Install all necessary Node modules**:
   ```bash
   npm install
   ```

3. **Spin up the Next.js development server**:
   ```bash
   npm run dev
   ```
   * *The dashboard is now running locally! Open your web browser and navigate to [http://localhost:3000](http://localhost:3000).*

---

## 💻 How to Use the Application

Once you have both servers running successfully:
1. Open your browser and navigate to **[http://localhost:3000](http://localhost:3000)**.
2. Drag and drop your retinal fundus image scan into the dotted file dropzone, or click it to browse files.
3. Click **"Analyze Image"**.
4. The system will send the file, compute probabilities, and show a beautiful dashboard with your **Primary Finding**, **Confidence Score**, **Detailed Classification Bar Chart**, and **Clinical Context** descriptors!

> 💡 **Pro Tip:** For the best results, use preprocessed, clean fundus images (centered, properly cropped). Generic internet images or non-retinal images will yield highly inaccurate predictions.

---

## 📊 Model Details & Classes

The core deep learning component is built on top of the **EfficientNet-B3** architecture loaded with ImageNet weights, refitted with a custom classification head, and fine-tuned on the multi-class **ODIR-5K** dataset. 

* **Train/Val/Test Split:** Patient-level split of **70% / 15% / 15%** (prevents data leakage between eye pairs of the same patient).
* **Final Model Performance:** Best Test Accuracy of **65.31%** with a Macro F1 score of **0.67**.

### 🏷️ Supported Eye Classification Conditions
The model categorizes retinal fundus scans into **6 classes**:
1. **Normal (N):** Healthy retinal fundus scans without detectable anomalies.
2. **Diabetic Retinopathy (D):** Scans exhibiting microaneurysms, hemorrhages, or exudates.
3. **Glaucoma (G):** Optic nerve head deterioration, marked by high cup-to-disc ratio.
4. **Cataract (C):** Scans showcasing cloudiness/opacification of the lens.
5. **Myopia (M):** Pathological changes associated with severe nearsightedness.
6. **Other (O):** Catch-all category. *(Note: Age-related Macular Degeneration (AMD) and Hypertensive Retinopathy are merged here due to insufficient distinct training samples)*.

---

## 🧪 Model Experiment History

The development path toward OcuSense V6 spanned multiple model iterations, architectures, regularization changes, and data split strategies:

| Experiment Version | Model Architecture / Strategy | Validation Acc (%) | Test Acc (%) | Status / Key Takeaway |
| :--- | :--- | :---: | :---: | :--- |
| **V1 (Baseline)** | EfficientNet-B3 Baseline | 47.93% | — | Baseline benchmark established. |
| **V2** | Heavy Regularization | 26.51% | — | Heavily underfit due to high dropout & weight decay. |
| **V3** | Clean Training Recipe | 61.69% | — | Optimized hyper-parameters & scheduled learning rate. |
| **V4** | Patient-Level Split Fix | — | 59.06% | Fixed data leakage issue. Accurate representation. |
| **V5** | Focal Loss Experiment | — | 57.19% | Attempt to fix class imbalance; performed slightly worse. |
| **V6 (FINAL)** | **6-Class Merge + Best Recipe** | **—** | **65.31%** | **Best combination. Finalized for deployment!** |
| **V7** | Overfitting Fix + TTA | — | 60.42% | Test-Time Augmentation resulted in minor loss of metrics. |

---

## 📡 API Endpoints

The Flask backend exposes a pair of simple REST API endpoints:

### `GET /health`
* **Description:** Check if the API is running and retrieve active model details.
* **Response format:**
  ```json
  {
    "status": "healthy",
    "model": "OcuSense V6"
  }
  ```

### `POST /predict`
* **Description:** Accepts an image payload and executes AI disease predictions.
* **Content-Type:** `multipart/form-data`
* **Payload parameters:**
  * `image`: File binary object (JPEG/PNG)
* **Response format:**
  ```json
  {
    "prediction": "Diabetic Retinopathy",
    "confidence": 88.42,
    "info": "Characterized by damage to the blood vessels of the light-sensitive tissue...",
    "all_probs": {
      "Normal": 5.12,
      "Diabetic Retinopathy": 88.42,
      "Glaucoma": 2.11,
      "Cataract": 0.95,
      "Myopia": 1.40,
      "Other": 2.00
    },
    "model": "OcuSense V6"
  }
  ```

---

## 🔍 Troubleshooting

Here are solutions to the most common configuration and execution issues:

#### 1. `ModuleNotFoundError: No module named 'flask'` (or similar libraries)
* **Reason:** Your active terminal session has not activated the virtual environment or the environment did not install dependencies.
* **Fix:** Make sure the `(backendvenv)` prefix is showing on your terminal command line. If not, re-run the activation script (`backendvenv\Scripts\Activate.ps1` or `source backendvenv/bin/activate`) and verify you ran `pip install` inside it.

#### 2. CORS Errors in Web Browser Console
* **Reason:** The Next.js frontend is running at port `3000` and attempting to request resources from Flask at port `5000`, which is blocked unless the backend permits cross-origin traffic.
* **Fix:** Ensure `flask-cors` is installed and initialized in `backend/main.py`. The initialization must have `CORS(app)` immediately under the app definition.

#### 3. High CPU Usage / Performance Stalls
* **Reason:** The model runs inference calculations on the CPU by default.
* **Fix:** While the models perform sufficiently on standard modern CPUs, ensure your input image is small (under 1-2MB) to reduce load. CUDA-supported GPUs will automatically be utilized if PyTorch is installed with GPU compatibility.

#### 4. Model outputs "Other" for almost all web images
* **Reason:** High-resolution camera photos or landscape images don't look like medical fundus scans and confuse the network.
* **Fix:** Always supply cropped, preprocessed retinal fundus scans. You can obtain reference sample files from the [ODIR-5K Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k).
