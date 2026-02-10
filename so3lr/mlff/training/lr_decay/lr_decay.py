def exponential(step: int, transition_steps: int, decay_factor: float):
    return decay_factor**(step/transition_steps)


def on_plateau(plateau_count: int, plateau_length: int, patience: int, decay_factor: float):
    """
    Given the n-th plateau, the length of the current plateau, the patience as well as the decay factor return
    a scaling factor for the learning rate, the updated plateau length, as well as the updated plateau count.

    Args:
        plateau_count ():
        plateau_length ():
        patience ():
        decay_factor ():

    Returns:

    """
    _plateau_length = plateau_length % patience
    _plateau_count = plateau_length // patience
    _cd = decay_factor ** plateau_count
    return _cd, _plateau_length, _plateau_count
