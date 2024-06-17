"""
Settings for minerva project.

"""
from pathlib import Path
from decouple import config, UndefinedValueError


class Config:
    try:
        # Build paths inside the project like this: BASE_DIR / 'subdir'.
        BASE_DIR = Path(__file__).resolve().parent.parent

        # SECURITY WARNING: keep the secret key used in production secret!
        SECRET_KEY = config('SECRET_KEY',
                            default='^*esc+si@gmnkv&9nl=fpf5d@l7%cy9@c@93673kk@i$9ql9*m')

        URL_VULCAN_API = config('URL_VULCAN_API')
    except UndefinedValueError as uverr:
        print(uverr)
    except Exception as ex:
        print(ex)

