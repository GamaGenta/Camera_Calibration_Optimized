import pickle

i = 2
while i <=4:
    file = 'stereo_cam1_cam' + str(i) + '.pkl'
    print(file)
    with open(file, 'rb') as f:
    #with open(f"stereo_cam1_cam{i}.pkl", 'rb') as f:

        data = pickle.load(f)

        print("Stereo data from 1 and" + str(i) + " \n")
        print(data)
        print("\n")

    i += 1
    
'''
i = 1
while i <=4:
    file = f"mono_cam{i}.pkl"
    print(file)
    with open(file, 'rb') as f:
    #with open(f"stereo_cam1_cam{i}.pkl", 'rb') as f:

        data = pickle.load(f)

        print("Mono Data for: " + str(i))
        print(data)
        print("\n")

    i += 1
    '''