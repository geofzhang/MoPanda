import pandas as pd
import xml.etree.ElementTree as ET
import os

def check_file(output_file):
    if os.path.exists(output_file):
        while True:
            response = input(
                f"The output file '{output_file}' already exists. Do you want to overwrite it? (Y/N): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                return False
            else:
                print("Invalid response. Please enter 'Y' or 'N'.")
    return True

class ColorCoding:
    def __init__(self):
        self.df = pd.DataFrame(columns=['name', 'label', 'color'])

    def litho_color(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()

        data = []
        for lithology in root.findall('lithology'):
            name = lithology.get('name')
            label = int(lithology.get('label'))
            color = lithology.get('color')

            data.append({'name': name, 'label': label, 'color': color})

        self.df = pd.concat([self.df, pd.DataFrame(data)], ignore_index=True)

        return self.df

    def name_to_label(self, lithology_name):
        if isinstance(lithology_name, (list, tuple)):
            return self.df.loc[self.df['name'].isin(lithology_name), 'label'].tolist()
        else:
            label = self.df.loc[self.df['name'] == lithology_name, 'label']
            return label.values[0] if not label.empty else None

    def label_to_name(self, lithology_label):
        if isinstance(lithology_label, (list, tuple)):
            return self.df.loc[self.df['label'].isin(lithology_label), 'name'].tolist()
        else:
            name = self.df.loc[self.df['label'] == lithology_label, 'name']
            return name.values[0] if not name.empty else None

    def label_to_color(self, lithology_label):
        if isinstance(lithology_label, (list, tuple)):
            return self.df.loc[self.df['label'].isin(lithology_label), 'color'].tolist()
        else:
            color = self.df.loc[self.df['label'] == lithology_label, 'color']
            return color.values[0] if not color.empty else None

    def name_to_color(self, lithology_name):
        if isinstance(lithology_name, (list, tuple)):
            return self.df.loc[self.df['name'].isin(lithology_name), 'color'].tolist()
        else:
            color = self.df.loc[self.df['name'] == lithology_name, 'color']
            return color.values[0] if not color.empty else None