import os
import glob
import json

CONTENT_DIR = "CORD-19-research-challenge"
section_dict = {}

headings_to_exclude_set = set()
with open("headings_to_exclude.txt") as f:
    for line in f:
        headings_to_exclude_set.add(line.strip())
print(headings_to_exclude_set)


def read_files():
    nr_of_sections = 0

    texts_list = []
    for dir in ["biorxiv_medrxiv, comm_use_subset", "custom_license", "noncomm_use_subset"]:
        path = os.path.join(CONTENT_DIR, dir, dir)
        files = glob.glob(path + "/*.json")
        for file in files:
            order_in_paper = 0
            with open(file) as f:
                data = json.load(f)
                paper_id = data["paper_id"]
                current_section = None
                for el in data["body_text"]:
                    if el["section"] != current_section:
                        current_section = el["section"].lower()
                        if current_section not in headings_to_exclude_set:
                        #print("\n" + current_section)
                            if current_section in section_dict:
                                section_dict[current_section] = section_dict[current_section] + 1
                            else:
                                section_dict[current_section] = 1
                            order_in_paper = order_in_paper + 1
                            tuple_to_append = (el["text"], current_section, paper_id, order_in_paper)
                            texts_list.append(tuple_to_append)
                            print(tuple_to_append)
                        #print("----")
                    #print("\n")
                    #print(el["text"])
                    nr_of_sections = nr_of_sections + 1
                    if nr_of_sections % 1000 == 0:
                        print(nr_of_sections)
                                    

    with open('headings.txt', 'w') as f:
        for (nr, k) in sorted([(nr, key) for (key, nr) in section_dict.items()], reverse=True):
            f.write(k + "\t" +  str(nr) + "\n")
    print(len(texts_list))
    
read_files()
