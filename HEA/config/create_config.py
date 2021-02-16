"""
Launch to create the configuration file of the project.
"""

import os
import configparser

if __name__ == '__main__':
    
    repo = os.getenv('ANAROOT')

    mode_create_config = None

    print('Welcome in the creation of the config file of HEA')

    get_repo = input("Choose the repository to be the value of the 'ANAROOT' environment variable ? (y/n) ")
    while get_repo not in ['y', 'n']:
        get_repo = input('y/n:')

    if get_repo=='n':
        repo = input("Absolute path of the repository of the project: ")

    name_project = input('Name of the project: ')


    config = configparser.ConfigParser()

    config['default'] = {}

    config['location'] = {}
    config['location']['ROOT'] = repo + '/'
    config['location']['OUT'] = config['location']['ROOT'] +  'output/'
    config['location']['PLOTS'] = config['location']['OUT'] + 'plots/'
    config['location']['TABLES'] = config['location']['OUT'] + 'tables/'
    config['location']['JSON'] = config['location']['OUT'] + 'json/'
    config['location']['PICKLE'] = config['location']['OUT'] + 'pickle/'
    config['location']['DEFINITION'] = config['location']['ROOT'] + f'{name_project}/definition.py'

    config['project'] = {}
    config['project']['name'] = name_project
    config['project']['text_plot'] = 'LHCb preliminary \n 2 fb$^{-1}$'

    config['fontsize'] = {}
    config['fontsize']['ticks'] = str(20)
    config['fontsize']['legend'] = str(20)
    config['fontsize']['label'] = str(25)
    config['fontsize']['text'] = str(25)
    config['fontsize']['annotation'] = str(15)


    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    print("The configuration file has been written.")

    if not os.path.isfile(config['location']['definition']):
        print(f"Beware, {config['location']['definition']} needs to created")
