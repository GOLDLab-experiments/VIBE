
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from time import time

from EmotionRecognition.EmotionRecognizer import EmotionRecognizer
from ObjectDetection.ObjectDetector import ObjectDetector

print("Loading model...")

# PALIGEMMA = 'google/paligemma-3b-ft-textcaps-448'
LLM_MODEL = 'meta-llama/Llama-3.2-1B-Instruct'
# LLM_MODEL = "facebook/opt-125m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LLM:
    def __init__(self, model_id=LLM_MODEL, device=DEVICE):
        print(f"Loading LLM model: {model_id} on device: {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.to(device)
        self.device = device
        print(f"LLM ({model_id}) loaded and moved to {device}.")
        # if self.tokenizer.pad_token is None:
        #    self.tokenizer.pad_token = self.tokenizer.eos_token
        self.objectDetection_model = ObjectDetector()
        self.emotionRecognition_model = EmotionRecognizer()


    def classify(self, frame, previous_output=None):

        object_description = ""

        # cv2_frame = None
        print("Starting Object detection...")
        objects = self.objectDetection_model.detect(frame)
        print("Objects detected:", objects)

        for obj, probability in objects:
            object_description += f"{obj} with probability {round(probability,3)}, "


        emotions_description = ""

        print("\nStarting Emotions classification...")
        _, emotions = self.emotionRecognition_model.detect(frame)

        for emotion, probability in emotions:
            prob = round(probability, 3)
            print(f"Emotion: {emotion}")
            emotions_description += f"{emotion} with probability {prob}, "

        print("Object description:", object_description)
        print("Emotions description:", emotions_description)

        # Round all probabilities in the emotions list
        rounded_emotions = [(emotion, round(probability, 3)) for emotion, probability in emotions]
        
        print("Rounded emotions:", rounded_emotions)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI model that classifies scenes based on detected objects and human emotions. "
                    "Given the following arrays of detected objects and emotions (each with probabilities), "
                    "classify the scene into one of these categories: Benign, Malicious, Authorized. "
                    "Respond with ONLY the category name, and nothing else. Do not explain your answer."

                )
            },
            {
                "role": "user",
                "content": (
                    f"Detected objects (with probabilities): {object_description}\n"
                    f"Detected emotions (with probabilities): {emotions_description}\n"
                    "What is the category? give me your reasoning <reasoning:> (Respond with only one word: Benign, Malicious, or Authorized)"
                )
            }
        ]

        if previous_output:
            messages.append({
                "role": "user",
                "content": f"Previous classification: {previous_output}"
            })

        print("\nMessages:", messages)

        # If your tokenizer supports chat templates, use it; otherwise, flatten messages to a prompt
        # if hasattr(self.tokenizer, "apply_chat_template"):
        #     print("In if has attribute")
        #     prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        # else:
        #     print("In else has attribute")
        #     # Fallback: simple concatenation
        prompt = "\n".join([msg["content"] for msg in messages])

        print("\nPrompt:\n", prompt)

        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        print("\nInputs processed and moved to", self.device, "for LLM.")
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=5,
            num_beams=1,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        output = generated_texts[0].strip()

        # output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print("\nLLM Output:\n", output)
        # Extract classification
        # predicted_class = "uncertain"

        # Only keep the first valid category word
        for word in output.split():
            if word.capitalize() in ["Benign", "Malicious", "Authorized"]:
                predicted_class = word.capitalize()
                break
        else:
            predicted_class = "uncertain"
        # predicted_class = output.partition("Classification:")[2].lstrip()

        print(f"\nClassification Result: {predicted_class}")
        return predicted_class



def main():

    # start = time()

    llm = LLM()
    image_description = ""

    
    # Use the LLM output as additional input to the LLM (demonstration)
    # refined_classification = llm.classify(image_description, previous_output=classification)
    # print(f"Refined Classification Result: {refined_classification}")

    # Capture image from camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press SPACE to capture an image, or ESC to exit.")
    while True:

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)

        if key % 256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            break

        elif key % 256 == 32:  # SPACE pressed
            image = frame.copy()

            classification = llm.classify(image)

            print("Classification:", classification)
            # Optionally, show detections on the image
            # for detection in detections:
            #     label, prob = detection
            #     cv2.putText(image, f"{label}: {prob}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # cv2.imshow("classification", image)
            # cv2.waitKey(0)
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # end = time()
    # print(f"Total time taken: {end - start:.2f} seconds")
    print("Processing complete.")

if __name__ == "__main__":
    main()
