stuff_classes = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road, route', 'bed', 'window ', 'grass',
                 'cabinet', 'sidewalk, pavement', 'person', 'earth, ground', 'door', 'table', 'mountain, mount',
                 'plant', 'curtain', 'chair', 'car', 'water', 'painting, picture', 'sofa', 'shelf', 'house', 'sea',
                 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock, stone',
                 'wardrobe, closet, press', 'lamp', 'tub', 'rail', 'cushion', 'base, pedestal, stand', 'box',
                 'column, pillar', 'signboard, sign', 'chest of drawers, chest, bureau, dresser', 'counter', 'sand',
                 'sink', 'skyscraper', 'fireplace', 'refrigerator, icebox', 'grandstand, covered stand', 'path',
                 'stairs', 'runway', 'case, display case, showcase, vitrine',
                 'pool table, billiard table, snooker table', 'pillow', 'screen door, screen', 'stairway, staircase',
                 'river', 'bridge, span', 'bookcase', 'blind, screen', 'coffee table',
                 'toilet, can, commode, crapper, pot, potty, stool, throne', 'flower', 'book', 'hill', 'bench',
                 'countertop', 'stove', 'palm, palm tree', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
                 'arcade machine', 'hovel, hut, hutch, shack, shanty', 'bus', 'towel', 'light', 'truck', 'tower',
                 'chandelier', 'awning, sunshade, sunblind', 'street lamp', 'booth', 'tv', 'plane', 'dirt track',
                 'clothes', 'pole', 'land, ground, soil', 'bannister, banister, balustrade, balusters, handrail',
                 'escalator, moving staircase, moving stairway', 'ottoman, pouf, pouffe, puff, hassock', 'bottle',
                 'buffet, counter, sideboard', 'poster, posting, placard, notice, bill, card', 'stage', 'van', 'ship',
                 'fountain', 'conveyer belt, conveyor belt, conveyer, conveyor, transporter', 'canopy',
                 'washer, automatic washer, washing machine', 'plaything, toy', 'pool', 'stool', 'barrel, cask',
                 'basket, handbasket', 'falls', 'tent', 'bag', 'minibike, motorbike', 'cradle', 'oven', 'ball',
                 'food, solid food', 'step, stair', 'tank, storage tank', 'trade name', 'microwave', 'pot', 'animal',
                 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket, cover', 'sculpture', 'hood, exhaust hood',
                 'sconce', 'vase', 'traffic light', 'tray', 'trash can', 'fan', 'pier', 'crt screen', 'plate',
                 'monitor', 'bulletin board', 'shower', 'radiator', 'glass, drinking glass', 'clock', 'flag']

stuff_colors = [[120, 120, 120],
                [180, 120, 120],
                [6, 230, 230],
                [80, 50, 50],
                [4, 200, 3],
                [120, 120, 80],
                [140, 140, 140],
                [204, 5, 255],
                [230, 230, 230],
                [4, 250, 7],
                [224, 5, 255],
                [235, 255, 7],
                [150, 5, 61],
                [120, 120, 70],
                [8, 255, 51],
                [255, 6, 82],
                [143, 255, 140],
                [204, 255, 4],
                [255, 51, 7],
                [204, 70, 3],
                [0, 102, 200],
                [61, 230, 250],
                [255, 6, 51],
                [11, 102, 255],
                [255, 7, 71],
                [255, 9, 224],
                [9, 7, 230],
                [220, 220, 220],
                [255, 9, 92],
                [112, 9, 255],
                [8, 255, 214],
                [7, 255, 224],
                [255, 184, 6],
                [10, 255, 71],
                [255, 41, 10],
                [7, 255, 255],
                [224, 255, 8],
                [102, 8, 255],
                [255, 61, 6],
                [255, 194, 7],
                [255, 122, 8],
                [0, 255, 20],
                [255, 8, 41],
                [255, 5, 153],
                [6, 51, 255],
                [235, 12, 255],
                [160, 150, 20],
                [0, 163, 255],
                [140, 140, 140],
                [250, 10, 15],
                [20, 255, 0],
                [31, 255, 0],
                [255, 31, 0],
                [255, 224, 0],
                [153, 255, 0],
                [0, 0, 255],
                [255, 71, 0],
                [0, 235, 255],
                [0, 173, 255],
                [31, 0, 255],
                [11, 200, 200],
                [255, 82, 0],
                [0, 255, 245],
                [0, 61, 255],
                [0, 255, 112],
                [0, 255, 133],
                [255, 0, 0],
                [255, 163, 0],
                [255, 102, 0],
                [194, 255, 0],
                [0, 143, 255],
                [51, 255, 0],
                [0, 82, 255],
                [0, 255, 41],
                [0, 255, 173],
                [10, 0, 255],
                [173, 255, 0],
                [0, 255, 153],
                [255, 92, 0],
                [255, 0, 255],
                [255, 0, 245],
                [255, 0, 102],
                [255, 173, 0],
                [255, 0, 20],
                [255, 184, 184],
                [0, 31, 255],
                [0, 255, 61],
                [0, 71, 255],
                [255, 0, 204],
                [0, 255, 194],
                [0, 255, 82],
                [0, 10, 255],
                [0, 112, 255],
                [51, 0, 255],
                [0, 194, 255],
                [0, 122, 255],
                [0, 255, 163],
                [255, 153, 0],
                [0, 255, 10],
                [255, 112, 0],
                [143, 255, 0],
                [82, 0, 255],
                [163, 255, 0],
                [255, 235, 0],
                [8, 184, 170],
                [133, 0, 255],
                [0, 255, 92],
                [184, 0, 255],
                [255, 0, 31],
                [0, 184, 255],
                [0, 214, 255],
                [255, 0, 112],
                [92, 255, 0],
                [0, 224, 255],
                [112, 224, 255],
                [70, 184, 160],
                [163, 0, 255],
                [153, 0, 255],
                [71, 255, 0],
                [255, 0, 163],
                [255, 204, 0],
                [255, 0, 143],
                [0, 255, 235],
                [133, 255, 0],
                [255, 0, 235],
                [245, 0, 255],
                [255, 0, 122],
                [255, 245, 0],
                [10, 190, 212],
                [214, 255, 0],
                [0, 204, 255],
                [20, 0, 255],
                [255, 255, 0],
                [0, 153, 255],
                [0, 41, 255],
                [0, 255, 204],
                [41, 0, 255],
                [41, 255, 0],
                [173, 0, 255],
                [0, 245, 255],
                [71, 0, 255],
                [122, 0, 255],
                [0, 255, 184],
                [0, 92, 255],
                [184, 255, 0],
                [0, 133, 255],
                [255, 214, 0],
                [25, 194, 194],
                [102, 255, 0],
                [92, 0, 255]]

# for idx, color in enumerate(stuff_colors):
#     print(f"{idx}: {color},")
label2colors = {
    0: [120, 120, 120],
    1: [180, 120, 120],
    2: [6, 230, 230],
    3: [80, 50, 50],
    4: [4, 200, 3],
    5: [120, 120, 80],
    6: [140, 140, 140],
    7: [204, 5, 255],
    8: [230, 230, 230],
    9: [4, 250, 7],
    10: [224, 5, 255],
    11: [235, 255, 7],
    12: [150, 5, 61],
    13: [120, 120, 70],
    14: [8, 255, 51],
    15: [255, 6, 82],
    16: [143, 255, 140],
    17: [204, 255, 4],
    18: [255, 51, 7],
    19: [204, 70, 3],
    20: [0, 102, 200],
    21: [61, 230, 250],
    22: [255, 6, 51],
    23: [11, 102, 255],
    24: [255, 7, 71],
    25: [255, 9, 224],
    26: [9, 7, 230],
    27: [220, 220, 220],
    28: [255, 9, 92],
    29: [112, 9, 255],
    30: [8, 255, 214],
    31: [7, 255, 224],
    32: [255, 184, 6],
    33: [10, 255, 71],
    34: [255, 41, 10],
    35: [7, 255, 255],
    36: [224, 255, 8],
    37: [102, 8, 255],
    38: [255, 61, 6],
    39: [255, 194, 7],
    40: [255, 122, 8],
    41: [0, 255, 20],
    42: [255, 8, 41],
    43: [255, 5, 153],
    44: [6, 51, 255],
    45: [235, 12, 255],
    46: [160, 150, 20],
    47: [0, 163, 255],
    48: [140, 140, 140],
    49: [250, 10, 15],
    50: [20, 255, 0],
    51: [31, 255, 0],
    52: [255, 31, 0],
    53: [255, 224, 0],
    54: [153, 255, 0],
    55: [0, 0, 255],
    56: [255, 71, 0],
    57: [0, 235, 255],
    58: [0, 173, 255],
    59: [31, 0, 255],
    60: [11, 200, 200],
    61: [255, 82, 0],
    62: [0, 255, 245],
    63: [0, 61, 255],
    64: [0, 255, 112],
    65: [0, 255, 133],
    66: [255, 0, 0],
    67: [255, 163, 0],
    68: [255, 102, 0],
    69: [194, 255, 0],
    70: [0, 143, 255],
    71: [51, 255, 0],
    72: [0, 82, 255],
    73: [0, 255, 41],
    74: [0, 255, 173],
    75: [10, 0, 255],
    76: [173, 255, 0],
    77: [0, 255, 153],
    78: [255, 92, 0],
    79: [255, 0, 255],
    80: [255, 0, 245],
    81: [255, 0, 102],
    82: [255, 173, 0],
    83: [255, 0, 20],
    84: [255, 184, 184],
    85: [0, 31, 255],
    86: [0, 255, 61],
    87: [0, 71, 255],
    88: [255, 0, 204],
    89: [0, 255, 194],
    90: [0, 255, 82],
    91: [0, 10, 255],
    92: [0, 112, 255],
    93: [51, 0, 255],
    94: [0, 194, 255],
    95: [0, 122, 255],
    96: [0, 255, 163],
    97: [255, 153, 0],
    98: [0, 255, 10],
    99: [255, 112, 0],
    100: [143, 255, 0],
    101: [82, 0, 255],
    102: [163, 255, 0],
    103: [255, 235, 0],
    104: [8, 184, 170],
    105: [133, 0, 255],
    106: [0, 255, 92],
    107: [184, 0, 255],
    108: [255, 0, 31],
    109: [0, 184, 255],
    110: [0, 214, 255],
    111: [255, 0, 112],
    112: [92, 255, 0],
    113: [0, 224, 255],
    114: [112, 224, 255],
    115: [70, 184, 160],
    116: [163, 0, 255],
    117: [153, 0, 255],
    118: [71, 255, 0],
    119: [255, 0, 163],
    120: [255, 204, 0],
    121: [255, 0, 143],
    122: [0, 255, 235],
    123: [133, 255, 0],
    124: [255, 0, 235],
    125: [245, 0, 255],
    126: [255, 0, 122],
    127: [255, 245, 0],
    128: [10, 190, 212],
    129: [214, 255, 0],
    130: [0, 204, 255],
    131: [20, 0, 255],
    132: [255, 255, 0],
    133: [0, 153, 255],
    134: [0, 41, 255],
    135: [0, 255, 204],
    136: [41, 0, 255],
    137: [41, 255, 0],
    138: [173, 0, 255],
    139: [0, 245, 255],
    140: [71, 0, 255],
    141: [122, 0, 255],
    142: [0, 255, 184],
    143: [0, 92, 255],
    144: [184, 255, 0],
    145: [0, 133, 255],
    146: [255, 214, 0],
    147: [25, 194, 194],
    148: [102, 255, 0],
    149: [92, 0, 255],
}
# print(len(stuff_colors))
# print(len(stuff_classes))
