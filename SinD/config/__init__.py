from pathlib import Path

# path to project root
ROOT = Path(__file__).parent.parent.absolute()

def get_dataset_path():
    '''
    Returns:
    ---
    - path to dataset folder
    '''
    return ROOT.joinpath('data')

def get_map_path():
    '''
    Returns:
    ---
    - path to intersection map file
    '''
    return f'{get_dataset_path()}/map_relink_law_save.osm'