import json
import os

OUTPUT_DIR = "for_annotation"
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

NO_MENTION = os.path.join(OUTPUT_DIR, "NoMention")
if not os.path.exists(NO_MENTION):
    os.mkdir(NO_MENTION)
    
MENTION = os.path.join(OUTPUT_DIR, "Mention")
if not os.path.exists(MENTION):
    os.mkdir(MENTION)

id_text_dict = {}
id_base_name_dict = {}

def read_t2t():
    model = "topics2themes_exports_folder_created_by_system/5e936908cfb357ff7689298a_model.json"
    with open(model) as f:
        model_data = json.load(f)
        for el in model_data["topic_model_output"]["documents"]:
            id_text_dict[str(el["id"])] = el["text"]
            id_base_name_dict[str(el["id"])] = el["base_name"]
            
            
    labels = "topics2themes_exports_folder_created_by_system/5e936908cfb357ff7689298a_user_defined_label.json"
    with open(labels) as f:
        labels_data = json.load(f)
        for el in labels_data:
            text_id = str(el["text_id"])
            user_defined_label = el["user_defined_label"]
            base_name = id_base_name_dict[text_id]
            text = id_text_dict[text_id]
            
            output_file = os.path.join(OUTPUT_DIR, user_defined_label, base_name)
            with open(output_file, "w") as w:
                w.write(text)
            
            if user_defined_label == "Mention":
                ann_file = os.path.join(OUTPUT_DIR, user_defined_label, base_name.replace(".txt", ".ann"))
                with open(ann_file, "w") as w:
                    w.write("")
                    

    
read_t2t()

    
