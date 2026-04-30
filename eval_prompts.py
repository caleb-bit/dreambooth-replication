PROMPTS = [
    "a {identifier} {class_name} in the jungle",
    "a {identifier} {class_name} on the beach",
    "a {identifier} {class_name} with a city in the background",
    "a {identifier} {class_name} in the snow",
    "a {identifier} {class_name} on top of a mountain",
    "a {identifier} {class_name} in a park",
    "a {identifier} {class_name} in a forest",
    "a {identifier} {class_name} in a museum",
    "a {identifier} {class_name} in a swimming pool",
    "a {identifier} {class_name} on the moon",
    "a {identifier} {class_name} in the Eiffel Tower",
    "a {identifier} {class_name} in the Grand Canyon",
    "a {identifier} {class_name} in Times Square",
    "a {identifier} {class_name} in front of the Colosseum",
    "a {identifier} {class_name} on a boat",
    "a {identifier} {class_name} on a bicycle",
    "a {identifier} {class_name} in a bakery",
    "a {identifier} {class_name} in a library",
    "a {identifier} {class_name} at the beach during sunset",
    "a {identifier} {class_name} in a garden",
    "a {identifier} {class_name} on a rooftop",
    "a {identifier} {class_name} in a coffee shop",
    "a {identifier} {class_name} in the rain",
    "a {identifier} {class_name} in front of a fireplace",
    "a {identifier} {class_name} in an art gallery",
]


def fill_specific(template: str, class_name: str, identifier: str = "sks") -> str:
    return template.format(identifier=identifier, class_name=class_name).strip()


def fill_general(template: str, class_name: str) -> str:
    return template.format(identifier="", class_name=class_name).replace("  ", " ").strip()
