import pandas as pd

df = pd.read_csv("all_annotations.csv")

movement = "{:02d}_{} ".format(df["no"][0], df["mov"][0])
movement_list = []
phrase_list = []
begin_phrase_index = 0
end_phrase_index = 0
last_index = len(df) - 1

def phrase_str(begin_phrase_index, end_phrase_index):
    return "{}:{}".format(begin_phrase_index, end_phrase_index) 

def movement_str(phrase_list, movement):
    return movement + " ".join(phrase_list)


for i, row in df.iterrows():
    if row["phraseend"]:
        end_phrase_index = i + 1
        phrase_list.append(phrase_str(begin_phrase_index, end_phrase_index))
        begin_phrase_index = end_phrase_index

    if i == last_index:
        if end_phrase_index != i + 1:
            end_phrase_index = i + 1
            phrase_list.append(phrase_str(begin_phrase_index, end_phrase_index))
            begin_phrase_index = end_phrase_index

        movement_list.append(movement_str(phrase_list, movement))
        phrase_list = []
        break

    current_movement = "{:02d}_{} ".format(row["no"], row["mov"])
    if current_movement != movement:
        # end of movement but not end of phrase
        if end_phrase_index != i:
            end_phrase_index = i 
            phrase_list.append(phrase_str(begin_phrase_index, end_phrase_index))
            begin_phrase_index = end_phrase_index

        movement_list.append(movement_str(phrase_list, movement))
        phrase_list = []
        movement = current_movement

with open("phrases.txt", "w") as f:
    f.write("\n".join(movement_list))
