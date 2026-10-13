"""Build the offline concept comparison from explicit calculated validation results."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write_redesign_viewer(output=None):
    output = output or HERE/'workspace_payload/redesign_options.html'
    report = json.loads((HERE/'workspace_payload/redesign_validation.json').read_text())
    assert report['calculation_checks_passed']
    html = (HERE/'payload_viewer/redesign_template.html').read_text()
    replacements = {
        '__REDESIGN_DATA__': json.dumps(report, separators=(',', ':'), allow_nan=False).replace('</', '<\\/'),
        '__REDESIGN_CSS__': (HERE/'payload_viewer/redesign.css').read_text(),
        '__REDESIGN_JS__': (HERE/'payload_viewer/redesign.js').read_text(),
    }
    for key, value in replacements.items():
        assert key in html, key
        html = html.replace(key, value)
    output.write_text(html)
    print(f'Built {output} ({output.stat().st_size/1e6:.1f} MB, offline)')
    return output


if __name__ == '__main__':
    write_redesign_viewer()
