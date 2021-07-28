with open("../models/test.txt", "w") as param_file:
    print({"a": 123}, file=param_file)
    print("Random", file=param_file)
    print({"b": 22}, file=param_file)