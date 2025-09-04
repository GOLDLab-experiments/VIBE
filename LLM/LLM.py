import cv2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from EmotionRecognition.EmotionRecognizer import EmotionRecognizer
from ObjectDetection.ObjectDetector import ObjectDetector
from skeleton.skeleton import SkeletonDetector

print("Loading model...")

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
        self.objectDetection_model = ObjectDetector(model_key="hf_helmet") # change object model here if needed
        self.emotionRecognition_model = EmotionRecognizer()
        self.skeletonDetector = SkeletonDetector()


    def classify(self, frame):
        # Object detection
        print("Starting Object detection...")
        objects = self.objectDetection_model.detect(frame)
        print("Objects detected:", objects)
        object_description = ", ".join(f"{obj} with probability {round(prob, 3)}" for obj, prob in objects)

        # Emotion recognition
        print("\nStarting Emotions classification...")
        _, emotions = self.emotionRecognition_model.detect(frame)
        print("Emotions detected:", emotions)
        emotions_description = emotions[0][0]

        # Skeleton detection
        print("\nStarting Skeleton detection...")
        skeleton_landmarks = self.skeletonDetector.detect(frame)
        if skeleton_landmarks is not None:
            skeleton_description = ", ".join(f"{landmark} with probability {prob}" for (landmark, prob) in skeleton_landmarks)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that classifies scenes based on detected objects, human emotions, and skeleton findings. "
                    "Given the detected objects, emotions, and skeleton findings - each with probabilities, "
                    "classify the scene into one of these categories: Benign, Malicious, Authorized. "
                    "The default classification should be Benign unless the objects, emotions, or skeleton suggest otherwise. "
                    "If the objects are common and non-threatening, classify as Benign. "
                    "If you detect objects or attire associated with work or safety environments (such as helmet, safety vest, uniform, badge, stethoscope, or other professional equipment), classify as Authorized. "
                    "First, output your reasoning inside <Reasoning> tags, e.g. <Reasoning> ... </Reasoning>. "
                    "Then, on a new line, output ONLY the category name (Benign, Malicious, or Authorized)."
                )
            },
            # FEW-SHOT EXAMPLES
            {
                "role": "user",
                "content": (
                    "Detected objects: Man with probability 0.603, Human face with probability 0.355, Clothing with probability 0.301, Drink with probability 0.221, \n"
                    "Detected emotions: happy \n"
                    "Skeleton findings: left_shoulder with probability 0.92, right_shoulder with probability 0.91, nose with probability 0.95\n"
                    "What is the category? Please provide your reasoning in <Reasoning> tags, then output only the category name on a new line.\n"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "<Reasoning> The detected emotion is happy, the objects are common and non-threatening, and the skeleton findings show a relaxed posture. This suggests a safe and non-threatening situation. </Reasoning>\n"
                    "Benign\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "Detected objects: helmet with probability 0.819, \n"
                    "Detected emotions: neutral \n"
                    "Skeleton findings: left_shoulder with probability 0.89, right_shoulder with probability 0.88, left_hip with probability 0.87\n"
                    "What is the category? Please provide your reasoning in <Reasoning> tags, then output only the category name on a new line.\n"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "<Reasoning> The presence of a helmet indicates a safety or work environment, meaning the person is Authoized. </Reasoning>\n"
                    "Authorized\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "Detected objects: Knife with probability 0.61, Scissors with probability 0.205, \n"
                    "Detected emotions: angry \n"
                    "Skeleton findings: left_wrist with probability 0.93, right_wrist with probability 0.92, left_elbow with probability 0.91\n"
                    "What is the category? Please provide your reasoning in <Reasoning> tags, then output only the category name on a new line.\n"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "<Reasoning> The presence of a knife and scissors and an angry emotion suggests a potentially dangerous and malicious situation. The skeleton findings may indicate raised arms, which could be threatening. </Reasoning>\n"
                    "Malicious\n"
                )
            },
            # END FEW-SHOT EXAMPLES
            {
                "role": "user",
                "content": (
                    f"Detected objects: {object_description}\n"
                    f"Detected emotions: {emotions_description}\n"
                    f"Skeleton findings: {skeleton_description}\n"
                    "What is the category? Please provide your reasoning in <Reasoning> tags, then output only the category name on a new line."
                )
            }
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False
            )
        except Exception:
            prompt = "\n".join([msg["content"] for msg in messages])

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
    
    def fast_classify(self, frame):
        # extract the features, and use predefined rules to classify
        objects = self.objectDetection_model.detect(frame)
        print("Objects detected:", objects)
        _, emotions = self.emotionRecognition_model.detect(frame)
        skeleton_landmarks = self.skeletonDetector.detect(frame)

        has_helmet = False
        # helmet -> Authorized
        for obj, prob in objects:
            if (obj.lower() == "helmet" and prob > 0.2) or (obj.lower() == "with helmet" and prob > 0.4):
                has_helmet = True
            
        # Weapon + angry -> Malicious
        weapons = {"knife", "scissors", "gun", "pistol"}
        has_weapon = any(label.lower() in weapons for (label, _p) in objects)

        # detect raised elbows (elbow above respective shoulder) treat as weapon equivalent
        raised_arm = False
        if skeleton_landmarks:
            # build quick lookup: name -> (prob, (x,y) or None)
            lm = {}
            for entry in skeleton_landmarks:
                # entry could be (name, prob) or (name, prob, (x,y))
                if len(entry) >= 3 and isinstance(entry[2], tuple):
                    name, p, coord = entry[0], entry[1], entry[2]
                else:
                    name, p, coord = entry[0], entry[1], None
                lm[name.lower()] = (p, coord)
            # check left
            if 'left_elbow' in lm and 'left_shoulder' in lm:
                le_p, le_coord = lm['left_elbow']
                ls_p, ls_coord = lm['left_shoulder']
                if le_coord and ls_coord and le_coord[1] < ls_coord[1]:  # smaller y => higher in image
                    raised_arm = True
            # check right
            if 'right_elbow' in lm and 'right_shoulder' in lm:
                re_p, re_coord = lm['right_elbow']
                rs_p, rs_coord = lm['right_shoulder']
                if re_coord and rs_coord and re_coord[1] < rs_coord[1]:
                    raised_arm = True

            has_weapon = has_weapon or raised_arm
        
        angry = False
        for emo, p in emotions:
            if emo.lower() == "anger" and p > 0.4:
                angry = True
        
        match (has_helmet, has_weapon, angry):
            case (True, _, _):
                return "Authorized", "Detected helmet with high probability."
            case (_, True, True):
                return "Malicious", "Detected weapon and angry emotion."
            case (_, True, False):
                return "Malicious", "Detected weapon."
            case (_, False, True):
                return "Malicious", "Detected angry emotion."
            case _:
                return "Benign", "No threats detected."

def main():

    llm = LLM()
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

            break

    cap.release()
    cv2.destroyAllWindows()

    print("Processing complete.")

if __name__ == "__main__":
    main()
