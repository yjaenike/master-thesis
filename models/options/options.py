from hydra import initialize, compose
import hydra


def is_notebook():
    ''' Checks if the current executed file is a notebook '''
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def get_options(file_name="options.yaml"):
    ''' Loads a options file if its a notebook, otherwise get the command line args and put them into the opt '''
    args = {}
    @hydra.main(config_path="conf", config_name="config")
    def command_line_cfg(cfg):
        cfg = dict(cfg)
        args.update(cfg)

    if is_notebook():
        with initialize(config_path="./"):
            args = compose(config_name=file_name, overrides=[])
    else:
        command_line_cfg()
    return dict(args)


def update_options(options, file_name="update_options.yaml"):
    '''Overwrites the options dict with the new updated options'''
    with initialize(config_path="./"):
        args = compose(config_name=file_name, overrides=[])
            
    new_options = dict(args)
    
    for key, value in new_options.items():
        options[key] = value
        
    return options

def print_options(opt):
    for key, value in opt.items():
        if key[0]=="-":
            print("\n\033[95m\033[1m","{:}".format(key),"\033[0m: ",value)
        else:
            print("\033[93m\033[1m","{:}".format(key),"\033[0m: ",value)

#TODO: Maybe turn the options into a Calls so they can be accessed with the options.X notation.
    
