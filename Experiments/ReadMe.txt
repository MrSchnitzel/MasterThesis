User of ExPlotter:
python ExPlotter.py 0 [x-Range] [polt pattern] [folders with csv's] 0
eg:
    python ExPlotter.py 0 4000 'a ba dir' /home/master/.opt/Thesis/tum-thesis-latex/Experiments/SplitHiddenPositive/Test/aTestM2+ /home/master/.opt/Thesis/tum-thesis-latex/Experiments/SplitHiddenPositive/Training/m2+training 0
    This plots 'Angle, BodyAngle and direction' in one plot and shows the last 4000 entries for the M2 Training and Test


[polt pattern]
        "wbl": Weights Body Left
        "wbr": Weights Body Right
        
        "whl": Weights Head Left
        "whr": Weights Head Right
        
        "wx": Weights hidden Layer
        "dir": direction Body
        "dirh": direction Head

        "a": angle  
        "ar": raw angle
        
        "ba": body angle
        "bar": raw body angle
        
        "dl": dope body left
        "dr": dope body right
        
        "dhl": dope head left
        "dhr":  Dope Head Right
        "dist": distance
Seperators:
    " " : same plot
    "_" : same row new plot
    "," : new plot new row
    ";" : new window