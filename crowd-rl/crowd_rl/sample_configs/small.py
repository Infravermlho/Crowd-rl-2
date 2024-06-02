dict_config = {
    "attendants": [
        {"id": "a1", "pos": {"x": 1, "y": 1}, "processing_time": [20, 50]},
        {"id": "b1", "pos": {"x": 3, "y": 1}, "processing_time": [10, 20]},
        {"id": "ab1", "pos": {"x": 7, "y": 5}, "processing_time": [10, 20]},
    ],
    "queues": [
        {
            "order": 0,
            "accepts": ["a"],
            "attendants": ["a1"],
            "wait_spots": [
                {"x": 1, "y": 4},
                {"x": 1, "y": 5},
                {"x": 1, "y": 6},
            ],
        },
        {
            "order": 0,
            "accepts": ["b"],
            "attendants": ["b1"],
            "wait_spots": [
                {"x": 3, "y": 4},
                {"x": 3, "y": 5},
                {"x": 3, "y": 6},
            ],
        },
        {
            "order": 1,
            "accepts": ["a", "b"],
            "attendants": ["ab1"],
            "wait_spots": [
                {"x": 5, "y": 5},
                {"x": 5, "y": 4},
                {"x": 5, "y": 3},
            ],
        },
    ],
    "entrances": [
        {"pos": {"x": 0, "y": 7}, "rate": 5, "accepts": ["a", "b", "c"]},
    ],
    "exits": [
        {"pos": {"x": 8, "y": 7}, "accepts": ["a", "b", "c"]},
    ],
    "groups": [
        {"name": "a", "amount": 3},
        {"name": "b", "amount": 3},
        {"name": "c", "amount": 3},
    ],
    "worldmap": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
}
