import os
import glob
import json
import csv

CONTENT_DIR = "../CORD-19-research-challenge"
section_dict = {}

headings_to_exclude_set = set()
with open("headings_to_exclude.txt") as f:
    for line in f:
        headings_to_exclude_set.add(line.strip())

meta_data_dict = {}
with open(os.path.join(CONTENT_DIR, 'metadata.csv')) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        id = row[0].strip()
        if ";" in id:
            id_list = [x.strip() for x in id.split(";")]
        else:
            id_list = [id]
        for id_el in id_list:
            meta_data_dict[id_el] = row

words_from_call = []
with open("smallwordsfromcall.txt") as f:
    for line in f:
        words_from_call.append(line.strip())

def read_files():
    nr_of_sections = 0
    found_words_dict = {}

    texts_list = []
    for dir in ["biorxiv_medrxiv, comm_use_subset", "custom_license", "noncomm_use_subset"]:
        path = os.path.join(CONTENT_DIR, dir, dir)
        files = glob.glob(path + "/*.json")
        for file in files:
            order_in_paper = 0
            with open(file) as f:
                data = json.load(f)
                paper_id = data["paper_id"].strip()
                meta_data = []
                if paper_id in meta_data_dict:
                    meta_data = meta_data_dict[paper_id]
                else:
                    print(paper_id + " not in metadata")
                    print(nr_of_sections)
                    print()
                    pass
                    
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
                                
                            add_text = False
                            found_words = []
                            for word in words_from_call:
                                if word in el["text"].lower():
                                    add_text = True
                                    found_words.append(word)
                                    if word in found_words_dict:
                                        found_words_dict[word] = found_words_dict[word] + 1
                                    else:
                                        found_words_dict[word] = 1
                            if add_text:
                                order_in_paper = order_in_paper + 1
                                list_to_append = [el["text"], current_section, paper_id, order_in_paper, dir]
                                list_to_append.extend(meta_data[1:])
                                list_to_append.append("/".join(found_words))
                                texts_list.append(list_to_append)
                            #print(list_to_append)
                        #print("----")
                    #print("\n")
                    #print(el["text"])
                    nr_of_sections = nr_of_sections + 1
                    if nr_of_sections % 10000 == 0:
                        print(nr_of_sections)
                                    

    with open('headings.txt', 'w') as f:
        for (nr, k) in sorted([(nr, key) for (key, nr) in section_dict.items()], reverse=True):
            f.write(k + "\t" +  str(nr) + "\n")
            
    with open('expanded_meta_data.txt', 'w') as f:
        for el in texts_list:
            f.write("\t".join([str(i) for i in el]) + "\n")
            
    for key, item in found_words_dict.items():
        print(str(key) + "\t" + str(item))
    
read_files()
