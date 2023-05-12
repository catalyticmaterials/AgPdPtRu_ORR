import time
from functools import wraps
import pickle

def uppercase_deco(function):
    """ makes string from function all uppercase letters """
    def wrapper():
        func = function()
        make_uppercase = func.upper()
        return make_uppercase
    return wrapper

def split_deco(function):
    """ splits string from function into list of strings"""
    def wrapper():
        func = function()
        splitted_string = func.split()
        return splitted_string
    return wrapper

def args_deco(function):
    """ prints the args and kwargs of a function """
    def wrapper(*args,**kwargs):
        print('The positional arguments are', args)
        print('The keyword arguments are', kwargs)
        return function(*args,**kwargs)
    return wrapper

def args_deco(*deco_args, **deco_kwargs):
    """ prints the args and kwargs of a function """
    def deco(function):
        def wrapper(*func_args,**func_kwargs):
            print('The positional arguments of the decorator are', deco_args)
            print('The keyword arguments of the decorator are', deco_kwargs)
            print('The positional arguments of the function are', func_args)
            print('The keyword arguments of the function are', func_kwargs)
            return function(*func_args,**func_kwargs)
        return wrapper
    return deco

def time_deco(function):
    """ prints the time a function took to run """
    @wraps(function)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        print(f"Function {function.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper

def pickle_deco(incl_timestamp = False):
    """ pickles the result of function """
    def deco(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            if incl_timestamp:
                s = f'{function.__name__}_{time.strftime("%H:%M:%S_%d.%m.%y")}'
            else: s = function.__name__
            with open(f'{s}.pkl', 'wb') as d:
                pickle.dump(result, d)
            print(f'Result of {function.__name__} saved as {s}.pkl')
            return result
        return wrapper
    return deco

def retry_deco(max_tries=5, delay_seconds=1):
    """ retries function a specified number of times """
    def deco(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            tries = 0
            while tries < max_tries:
                try:
                    return function(*args, **kwargs)
                except Exception as e:
                    tries += 1
                    if tries == max_tries:
                        raise e
                    print(f"{function.__name__} failed. Retry's left: {max_tries-tries}")
                    time.sleep(delay_seconds)
        return wrapper
    return deco
