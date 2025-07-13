import multiprocessing

manager = multiprocessing.Manager()
damage_store = manager.dict()
