import logging

import nest_asyncio
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv(override=True)
    from streamlit_ui import main

    nest_asyncio.apply()
    try:
        main()
    except Exception as e:
        logging.exception(e)
        raise e
