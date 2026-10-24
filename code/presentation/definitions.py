"""Display labels, reaction names and the roles of the two substrates."""
STRUCTURE_FEATURES = {"halogen next to ring N": ("Halogen beside a ring N", "Brc1ccccn1"),
                      "N–H": ("N–H in the aryl halide", "Brc1cccc2[nH]ncc12")}

STRUCTURE_SETS = {"first condition set": ("o", 4.6, True), "second condition set": ("D", 3.4, False)}

ROLE = {"Acylation": ("acid", "amine"), "Esterification": ("acid", "alcohol"),
        "Buchwald-Hartwig": ("aryl halide", "amine"), "SNAr": ("aryl halide", "amine"),
        "Chan-Lam": ("boronic acid", "amine"), "Reductive-amination": ("carbonyl", "amine"),
        "Suzuki": ("aryl halide", "boronic acid"), "Suzuki Bpin": ("aryl halide", "boronate"),
        "Heck": ("aryl bromide", "alkene"), "Sonogashira": ("aryl bromide", "alkyne"),
        "Hiyama": ("aryl halide", "silane"), "Kumada": ("aryl halide", "RMgX"),
        "Grignard": ("carbonyl", "RMgX"), "Ullmann": ("aryl bromide", "phenol"),
        "C-O competitive reactions": ("aryl bromide", "O-nucleophile")}

NAMES = {"Acylation": "Amide coupling", "Heck": "Heck", "Suzuki": "Suzuki–Miyaura",
         "Ullmann": "Ullmann ether", "Buchwald-Hartwig": "Buchwald–Hartwig",
         "Kumada": "Kumada", "Chan-Lam": "Chan–Lam", "SNAr": "S$_\\mathregular{N}$Ar",
         "Hiyama": "Hiyama", "Grignard": "Grignard addition", "C-O competitive reactions": "C–O coupling",
         "Reductive-amination": "Reductive amination", "Esterification": "Esterification",
         "Sonogashira": "Sonogashira", "Suzuki Bpin": "Suzuki–Miyaura (Bpin)"}

REACTION_NAMES = {'Acylation': 'Amide coupling', 'Buchwald-Hartwig': 'Buchwald–Hartwig amination', 'Chan-Lam': 'Chan–Lam coupling', 'Heck': 'Heck reaction', 'SNAr': 'S<sub>N</sub>Ar', 'Suzuki': 'Suzuki–Miyaura coupling', 'Esterification': 'Esterification', 'Grignard': 'Grignard addition', 'Kumada': 'Kumada coupling', 'Hiyama': 'Hiyama coupling', 'Sonogashira': 'Sonogashira coupling', 'Ullmann': 'Ullmann ether synthesis', 'C-O competitive reactions': 'C–O coupling', 'Reductive-amination': 'Reductive amination', 'Suzuki Bpin': 'Suzuki–Miyaura coupling (Bpin)'}

REACTIONS = ("Acylation", "Buchwald-Hartwig", "Chan-Lam", "Heck", "SNAr", "Suzuki")

REAGENT_COLUMNS = [f"reagent{i}_id" for i in range(1, 7)]
