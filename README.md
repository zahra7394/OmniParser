# OmniParser: Screen Parsing tool for Pure Vision Based GUI Agent

<p align="center">
  <img src="imgs/logo.png" alt="Logo">
</p>

[![arXiv](https://img.shields.io/badge/Paper-green)](https://arxiv.org/abs/2408.00203)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📢 [[Project Page](https://microsoft.github.io/OmniParser/)] [[Blog Post](https://www.microsoft.com/en-us/research/articles/omniparser-for-pure-vision-based-gui-agent/)] [[Models](https://huggingface.co/microsoft/OmniParser)] [huggingface space](https://huggingface.co/spaces/microsoft/OmniParser)

**OmniParser** is a comprehensive method for parsing user interface screenshots into structured and easy-to-understand elements, which significantly enhances the ability of GPT-4V to generate actions that can be accurately grounded in the corresponding regions of the interface. 

## News
- [2024/11/26] We release an updated version, OmniParser V1.5 which features 1) more fine grained/small icon detection, 2) prediction of whether each screen element is interactable or not. Examples in the demo.ipynb. 
- [2024/10] OmniParser was the #1 trending model on huggingface model hub (starting 10/29/2024). 
- [2024/10] Feel free to checkout our demo on [huggingface space](https://huggingface.co/spaces/microsoft/OmniParser)! (stay tuned for OmniParser + Claude Computer Use)
- [2024/10] Both Interactive Region Detection Model and Icon functional description model are released! [Hugginface models](https://huggingface.co/microsoft/OmniParser)
- [2024/09] OmniParser achieves the best performance on [Windows Agent Arena](https://microsoft.github.io/WindowsAgentArena/)! 

## Install 
Install environment:
```python
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

Then download the model ckpts files in: https://huggingface.co/microsoft/OmniParser, and put them under weights/, default folder structure is: weights/icon_detect, weights/icon_caption_florence, weights/icon_caption_blip2. 

Finally, convert the safetensor to .pt file. 
```python
python weights/convert_safetensor_to_pt.py
```

## Examples:
We put together a few simple examples in the demo.ipynb. 

## Gradio Demo
To run gradio demo, simply run:
```python
python gradio_demo.py
```

## Model Weights License
For the model checkpoints on huggingface model hub, please note that icon_detect model is under AGPL license since it is a license inherited from the original yolo model. And icon_caption_blip2 & icon_caption_florence is under MIT license. Please refer to the LICENSE file in the folder of each model: https://huggingface.co/microsoft/OmniParser.

## Features
### **FastAPI Integration**
A REST API using FastAPI is developed to process images uploaded by users, analyze them using OmniParser, and return both the processed data and a labeled image. This would enable remote interaction with the processing tool.

1. **Start the Server**  
   Launch the FastAPI server by running `app.py` on your server:  
   ```bash
   python app.py
   
2. Send a POST request to `http://<your-server-address>:5000/process`

**Output Format**: The API returns a structured list of detections, each represented as a list of strings in the format `["ID", "Type", "Text", "x1", "y1", "x2", "y2"].`
- ID: The unique identifier for each detected box.
- Type: The classification of the detected box, either "Text" or "icon".
- Text: The label or content associated with the detected box.
- x1, y1: The coordinates of the top-left corner of the detected box.
- x2, y2: The coordinates of the bottom-right corner of the detected box.

Here's a Python script that demonstrates how to send an image to the API and handle the response:

```python
import requests
import base64
from PIL import Image
from io import BytesIO
from os.path import basename

# API endpoint
url = "http://<your-server-address>:5000/process"

# Path to the image you want to send
image_path = ""
image_name = basename(image_path)

# Send the image as a POST request
with open(image_path, 'rb') as image_file:
    response = requests.post(
        url,
        files={'image': image_file},  # Send the image file here
        data={'img_name': image_name}  # Send additional form data here
    )

# Check the response
if response.status_code == 200:
    data = response.json()
    
    # Get the base64 image from the response
    base64_image = data['based64_image']

    # Decode the image
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data))

    # Display the image on your local machine
    image.show()

    print("structured_results:", data["structured_results"])
else:
    print(f"Failed to get a response. Status code: {response.status_code}")
```



## 📚 Citation
Our technical report can be found [here](https://arxiv.org/abs/2408.00203).
If you find our work useful, please consider citing our work:
```
@misc{lu2024omniparserpurevisionbased,
      title={OmniParser for Pure Vision Based GUI Agent}, 
      author={Yadong Lu and Jianwei Yang and Yelong Shen and Ahmed Awadallah},
      year={2024},
      eprint={2408.00203},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00203}, 
}
```
