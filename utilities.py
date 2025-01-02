import configparser


def get_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return dict(config.items("main"))