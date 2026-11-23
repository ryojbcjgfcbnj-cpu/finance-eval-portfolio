"""Copy computed scan and benchmark summaries into the portable dashboard."""

import json
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    report = here.parent / 'outputs' / 'market_scale' / 'scale_report.json'
    data = json.loads(report.read_text(encoding='utf-8'))
    (here / 'dist' / 'scale-snapshot.js').write_text(
        'window.marketScale = ' + json.dumps(data, ensure_ascii=False, allow_nan=False) + ';\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
