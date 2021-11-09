class ActionEnum(enumerate):
    """
    Enum for action space
    """
    BUY = 1
    SELL = -1
    HOLD = 0
    EXIT = 3

def form_action(action):
    """
    Form action from action space
    """
    r = 0.0
    if action <= -0.5:
        r =-(action + 0.5) * 2
        return ActionEnum.SELL, r, "SELL"
    elif action >= +0.5:
        r = (action - 0.5) * 2
        return ActionEnum.BUY, r, "BUY"   
    else:
        return ActionEnum.HOLD, r, "NIL"

def normalize(x,min, max):
    """
    Scale [min, max] to [0, 1]
    Normalize x to [min, max] 
    """
    return (x - min) / (max - min)    
    