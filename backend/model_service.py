import io
import os
import base64

import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "best_model_1024px.pth"
)

DEVICE = torch.device("cpu")

CLASS_LABELS = {
    0: "Grade 0",
    1: "Grade 1",
    2: "Grade 2",
    3: "Grade 3",
    4: "Grade 4"
}


model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 5)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


def predict_image(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    resized_image = image.resize((512, 512))

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)

        confidence, predicted_class = torch.max(
            probabilities, dim=1
        )

    predicted_class = int(predicted_class.item())
    confidence = float(confidence.item() * 100)

    # Grad-CAM visualization
    target_layers = [model.layer4[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=None
    )[0]

    rgb_image = np.array(resized_image).astype(np.float32) / 255.0

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    pil_visualization = Image.fromarray(visualization)

    buffer = io.BytesIO()
    pil_visualization.save(buffer, format="PNG")

    visualization_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")



    return {
        "class": predicted_class,
        "label": CLASS_LABELS[predicted_class],
        "confidence": round(confidence, 2),
        "visualization": (
            "data:image/png;base64,"
            + visualization_base64
        )
    }
