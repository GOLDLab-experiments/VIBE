
import cv2
import torch
import re
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer

from EmotionRecognition.EmotionRecognizer import EmotionRecognizer
from ObjectDetection.ObjectDetector import ObjectDetector

print("Loading model...")

# LLM_MODEL = 'meta-llama/Llama-3.2-1B-Instruct'
# LLM_MODEL = "facebook/opt-125m"
LLM_MODEL = "Qwen/Qwen3-0.6B"

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

        # Object detection
        print("Starting Object detection...")
        objects = self.objectDetection_model.detect(frame)
        print("Objects detected:", objects)
        object_description = ", ".join(
            f"{obj} with probability {round(prob, 3)}" for obj, prob in objects
        )

        # Emotion recognition
        print("\nStarting Emotions classification...")
        _, emotions = self.emotionRecognition_model.detect(frame)
        print("Emotions detected:", emotions)
        emotions_description = ", ".join(
            f"{emotion} with probability {prob}" for (emotion, prob) in emotions
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that classifies scenes based on detected objects and human emotions. "
                    "Given the detected objects and emotions - each with probabilities, "
                    "classify the scene into one of these categories: Benign, Malicious, Authorized. "
                    "The default classification should be Benign unless the objects or emotions suggest otherwise. "
                    "If the objects are common and non-threatening, classify as Benign. "
                    "First, output your reasoning inside <Reasoning> tags, e.g. <Reasoning> ... </Reasoning>. "
                    "Then, on a new line, output ONLY the category name (Benign, Malicious, or Authorized)."
                )
            },
            # FEW-SHOT EXAMPLES
            {
                "role": "user",
                "content": (
                    "Detected objects: Man with probability 0.603, Human face with probability 0.355, Clothing with probability 0.301, Drink with probability 0.221, \n"
                    "Detected emotions: happy with probability 0.52, neutral with probability 0.41, \n"
                    "What is the category? Please provide your reasoning in <Reasoning> tags, then output only the category name on a new line.\n"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "<Reasoning> The detected emotion is happy and neutral, and the objects are common and non-threatening. This suggests a safe and non-threatening situation. </Reasoning>\n"
                    "Benign\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "Detected objects: Pen with probability 0.382, Suit with probability 0.219, \n"
                    "Detected emotions: neutral with probability 0.60, \n"
                    "What is the category? Please provide your reasoning in <Reasoning> tags, then output only the category name on a new line.\n"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "<Reasoning> The detected objects are office tools, which are typically found in authorized environments like hospitals. The emotion is neutral, which is not concerning. </Reasoning>\n"
                    "Authorized\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "Detected objects: Knife with probability 0.61, Scissors with probability 0.205, \n"
                    "Detected emotions: angry with probability 0.691, neutral with probability 0.144, happy with probability 0.105, surprized with probability 0.037, sad with probability 0.023 \n"
                    "What is the category? Please provide your reasoning in <Reasoning> tags, then output only the category name on a new line.\n"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "<Reasoning> The presence of a knife and scissors and an angry emotion suggests a potentially dangerous and malicious situation. </Reasoning>\n"
                    "Malicious\n"
                )
            },
            # END FEW-SHOT EXAMPLES
            {
                "role": "user",
                "content": (
                    f"Detected objects: {object_description}\n"
                    f"Detected emotions: {emotions_description}\n"
                    "What is the category? Please provide your reasoning in <Reasoning> tags, then output only the category name on a new line."
                )
            }
        ]

        if previous_output:
            messages.append({
                "role": "user",
                "content": f"Previous classification: {previous_output}"
            })

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False
            )
        except Exception:
            prompt = "\n".join([msg["content"] for msg in messages])

        print("\nPrompt:\n", prompt)

        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        print("\nInputs processed and moved to", self.device, "for LLM.")
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        output = generated_texts[0].strip()

        print("\nLLM Output:\n", output)

         # Extract reasoning using startswith/endswith and get the last one
        reasoning = "Not detected"

        lines = output.splitlines()
        last_reasoning_idx = -1
        for idx, line in enumerate(lines):
            line = line.strip()
            if line.startswith("<Reasoning>") and line.endswith("</Reasoning>"):
                reasoning = line[len("<Reasoning>"):-len("</Reasoning>")].strip()
                last_reasoning_idx = idx

        print("\nExtracted Reasoning:\n", reasoning)

        # Find the first valid category after the last </Reasoning>
        predicted_class = "uncertain"
        if last_reasoning_idx != -1:
            search_lines = lines[last_reasoning_idx+1:]
        else:
            search_lines = lines
        for line in search_lines:
            for word in line.split():
                if word.capitalize() in {"Benign", "Malicious", "Authorized"}:
                    predicted_class = word.capitalize()
                    break
            if predicted_class != "uncertain":
                break

        return predicted_class, reasoning

def main():

    llm = LLM()

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

            classification, reasoning = llm.classify(image)

            print("\nClassification:", classification)
            print("\nReasoning:", reasoning)


            # Optionally, show detections on the image
            # for detection in detections:
            #     label, prob = detection
            #     cv2.putText(image, f"{label}: {prob}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # cv2.imshow("classification", image)
            # cv2.waitKey(0)
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Processing complete.")

if __name__ == "__main__":
    main()
