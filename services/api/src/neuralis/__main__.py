import os

import uvicorn


def main():
    uvicorn.run("neuralis.main:app", host=os.getenv("NEURALIS_HOST", "127.0.0.1"),
                port=int(os.getenv("NEURALIS_PORT", "8000")),
                log_level=os.getenv("NEURALIS_LOG_LEVEL", "info"), access_log=False)


if __name__ == "__main__":
    main()
