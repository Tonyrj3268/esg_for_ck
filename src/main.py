import logging

import nest_asyncio
from dotenv import load_dotenv

# # 設置基本配置
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(message)s",
#     filename="esg.log",  # 設置日誌文件名
#     filemode="a",
# )  # 追加模式寫入文件


if __name__ == "__main__":
    load_dotenv(override=True)
    from streamlit_ui import main

    nest_asyncio.apply()
    try:
        
        # idle_thread = threading.Thread(target=check_idle)
        # idle_thread.daemon = True
        # idle_thread.start()
        main()
    except Exception as e:
        logging.exception(e)
        raise e
