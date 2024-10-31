import json
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom


def json_to_xml(json_obj, root=None):
    if root is None:
        root = ET.Element("claim")

    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            child = ET.SubElement(root, key)
            json_to_xml(value, child)
    elif isinstance(json_obj, list):
        if not json_obj:
            root.text = ""
        else:
            for item in json_obj:
                json_to_xml(item, root)
    else:
        root.text = str(json_obj)

    return root


def prettify(elem):
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def convert_json_to_xml(input_file):
    # Read JSON file
    json_path = Path(input_file)
    with json_path.open("r") as json_file:
        json_data = json.load(json_file)

    # Convert JSON to XML
    root = json_to_xml(json_data)

    # Create pretty XML string
    xml_string = prettify(root)

    # Write XML to file
    output_file = json_path.with_suffix(".xml")
    with output_file.open("w") as xml_file:
        xml_file.write(xml_string)

    print(f"Converted {input_file} to {output_file}")


[convert_json_to_xml(file) for file in Path("charts").glob("*claim*")]
