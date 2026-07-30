"""Channel topology annotations in raw 1-based AlphaFold-model numbering.

Transmembrane boundaries are sequence-mapped from the reviewed UniProt records
P15387 (rat Kv2.1), Q14524 (human Nav1.5), and Q13936 (human Cav1.2).  The
actual WT AlphaFold construct sequence was globally aligned to the corresponding
UniProt sequence before transferring each curated feature.  This mapping step is
essential for the shortened Kv2.1 and Nav1.5 constructs.

UniProt transmembrane features describe the membrane-spanning core and do not
always include the complete helical cytosolic extension visible in a structure.
The Cav1.2 DI-S6 extension containing G402 and G406 is therefore shown
separately. Pore-helix and selectivity-filter boundaries without a dedicated
UniProt feature remain explicitly provisional.
"""

TOPOLOGY_PROVENANCE = {
    "kv21": {
        "uniprot": "P15387",
        "model_length": 600,
        "method": "global sequence mapping of reviewed UniProt TM features",
    },
    "nav15": {
        "uniprot": "Q14524",
        "model_length": 1572,
        "method": "global sequence mapping of reviewed UniProt TM features",
    },
    "cav12": {
        "uniprot": "Q13936",
        "model_length": 1685,
        "method": "global sequence mapping of reviewed UniProt TM features",
    },
}

TOPOLOGY = {
    "kv21": [
        {"label": "S1", "start": 185, "end": 206, "domain": "VSD", "confidence": "reviewed/mapped"},
        {"label": "S2", "start": 227, "end": 248, "domain": "VSD", "confidence": "reviewed/mapped"},
        {"label": "S3", "start": 258, "end": 278, "domain": "VSD", "confidence": "reviewed/mapped"},
        {"label": "S4", "start": 293, "end": 314, "domain": "VSD", "confidence": "reviewed/mapped"},
        {"label": "S4–S5", "start": 315, "end": 328, "domain": "linker", "confidence": "sequence-defined"},
        {"label": "S5", "start": 329, "end": 349, "domain": "pore", "confidence": "reviewed/mapped"},
        {"label": "P helix", "start": 357, "end": 372, "domain": "pore", "confidence": "provisional"},
        {"label": "SF", "start": 375, "end": 379, "domain": "filter", "confidence": "motif/mapped"},
        {"label": "S6", "start": 390, "end": 418, "domain": "pore", "confidence": "reviewed/mapped"},
    ],
    "nav15": [
        {"label": "DI S1", "start": 131, "end": 150, "domain": "DI", "confidence": "reviewed/mapped"},
        {"label": "DI S2", "start": 159, "end": 180, "domain": "DI", "confidence": "reviewed/mapped"},
        {"label": "DI S3", "start": 190, "end": 210, "domain": "DI", "confidence": "reviewed/mapped"},
        {"label": "DI S4", "start": 218, "end": 237, "domain": "DI", "confidence": "reviewed/mapped"},
        {"label": "DI S5", "start": 251, "end": 273, "domain": "DI", "confidence": "reviewed/mapped"},
        {"label": "DI S6", "start": 388, "end": 414, "domain": "DI", "confidence": "reviewed/mapped"},
        {"label": "DII S1", "start": 524, "end": 541, "domain": "DII", "confidence": "reviewed/mapped"},
        {"label": "DII S2", "start": 551, "end": 573, "domain": "DII", "confidence": "reviewed/mapped"},
        {"label": "DII S3", "start": 580, "end": 600, "domain": "DII", "confidence": "reviewed/mapped"},
        {"label": "DII S4", "start": 611, "end": 625, "domain": "DII", "confidence": "reviewed/mapped"},
        {"label": "DII S5", "start": 643, "end": 664, "domain": "DII", "confidence": "reviewed/mapped"},
        {"label": "DII S6", "start": 719, "end": 747, "domain": "DII", "confidence": "reviewed/mapped"},
        {"label": "DIII S1", "start": 888, "end": 909, "domain": "DIII", "confidence": "reviewed/mapped"},
        {"label": "DIII S2", "start": 921, "end": 943, "domain": "DIII", "confidence": "reviewed/mapped"},
        {"label": "DIII S3", "start": 953, "end": 975, "domain": "DIII", "confidence": "reviewed/mapped"},
        {"label": "DIII S4", "start": 982, "end": 1001, "domain": "DIII", "confidence": "reviewed/mapped"},
        {"label": "DIII S5", "start": 1015, "end": 1039, "domain": "DIII", "confidence": "reviewed/mapped"},
        {"label": "DIII S6", "start": 1130, "end": 1154, "domain": "DIII", "confidence": "reviewed/mapped"},
        {"label": "IFM", "start": 1169, "end": 1171, "domain": "III–IV linker", "confidence": "high"},
        {"label": "DIV S1", "start": 1213, "end": 1231, "domain": "DIV", "confidence": "reviewed/mapped"},
        {"label": "DIV S2", "start": 1243, "end": 1264, "domain": "DIV", "confidence": "reviewed/mapped"},
        {"label": "DIV S3", "start": 1274, "end": 1296, "domain": "DIV", "confidence": "reviewed/mapped"},
        {"label": "DIV S4", "start": 1304, "end": 1324, "domain": "DIV", "confidence": "reviewed/mapped"},
        {"label": "DIV S5", "start": 1335, "end": 1363, "domain": "DIV", "confidence": "reviewed/mapped"},
        {"label": "DIV S6", "start": 1430, "end": 1455, "domain": "DIV", "confidence": "reviewed/mapped"},
    ],
    "cav12": [
        {"label": "DI S1", "start": 125, "end": 143, "domain": "DI", "confidence": "reviewed/mapped"},
        {"label": "DI S2", "start": 159, "end": 179, "domain": "DI", "confidence": "reviewed/mapped"},
        {"label": "DI S3", "start": 189, "end": 209, "domain": "DI", "confidence": "reviewed/mapped"},
        {"label": "DI S4", "start": 233, "end": 251, "domain": "DI", "confidence": "reviewed/mapped"},
        {"label": "DI S5", "start": 269, "end": 290, "domain": "DI", "confidence": "reviewed/mapped"},
        {"label": "DI S6 core", "start": 381, "end": 401, "domain": "DI", "confidence": "reviewed/mapped"},
        {"label": "DI S6 extension", "start": 402, "end": 412, "domain": "DI", "confidence": "structure-guided"},
        {"label": "DII S1", "start": 525, "end": 543, "domain": "DII", "confidence": "reviewed/mapped"},
        {"label": "DII S2", "start": 555, "end": 575, "domain": "DII", "confidence": "reviewed/mapped"},
        {"label": "DII S3", "start": 587, "end": 606, "domain": "DII", "confidence": "reviewed/mapped"},
        {"label": "DII S4", "start": 616, "end": 634, "domain": "DII", "confidence": "reviewed/mapped"},
        {"label": "DII S5", "start": 654, "end": 673, "domain": "DII", "confidence": "reviewed/mapped"},
        {"label": "DII S6", "start": 726, "end": 745, "domain": "DII", "confidence": "reviewed/mapped"},
        {"label": "DIII S1", "start": 901, "end": 919, "domain": "DIII", "confidence": "reviewed/mapped"},
        {"label": "DIII S2", "start": 932, "end": 952, "domain": "DIII", "confidence": "reviewed/mapped"},
        {"label": "DIII S3", "start": 988, "end": 1006, "domain": "DIII", "confidence": "reviewed/mapped"},
        {"label": "DIII S4", "start": 1014, "end": 1032, "domain": "DIII", "confidence": "reviewed/mapped"},
        {"label": "DIII S5", "start": 1052, "end": 1071, "domain": "DIII", "confidence": "reviewed/mapped"},
        {"label": "DIII S6", "start": 1160, "end": 1181, "domain": "DIII", "confidence": "reviewed/mapped"},
        {"label": "DIV S1", "start": 1240, "end": 1261, "domain": "DIV", "confidence": "reviewed/mapped"},
        {"label": "DIV S2", "start": 1270, "end": 1291, "domain": "DIV", "confidence": "reviewed/mapped"},
        {"label": "DIV S3", "start": 1302, "end": 1321, "domain": "DIV", "confidence": "reviewed/mapped"},
        {"label": "DIV S4", "start": 1373, "end": 1391, "domain": "DIV", "confidence": "reviewed/mapped"},
        {"label": "DIV S5", "start": 1410, "end": 1430, "domain": "DIV", "confidence": "reviewed/mapped"},
        {"label": "DIV S6", "start": 1500, "end": 1524, "domain": "DIV", "confidence": "reviewed/mapped"},
    ],
}


def validate_topology() -> None:
    expected_lengths = {"kv21": 600, "nav15": 1572, "cav12": 1685}
    for channel, segments in TOPOLOGY.items():
        for segment in segments:
            start, end = int(segment["start"]), int(segment["end"])
            if start < 1 or end < start or end > expected_lengths[channel]:
                raise ValueError(f"Invalid provisional topology segment: {channel} {segment}")


validate_topology()
