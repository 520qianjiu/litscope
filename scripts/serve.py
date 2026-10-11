import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn


def main() -> None:
    host = os.environ.get('HOST') or '0.0.0.0'
    port = int(os.environ.get('PORT') or 8000)
    print('LitScope serving on ' + host + ':' + str(port), flush=True)
    uvicorn.run('app.api.app:app', host=host, port=port, log_level='info')


if __name__ == '__main__':
    main()
