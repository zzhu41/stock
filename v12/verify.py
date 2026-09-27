"""Verify the completed V12 research bundle; optional one-time release sealing."""
import argparse
import json
from pathlib import Path

from .data import BASE, sha, dump, stamp, protect


def require(value,message):
    if not value:
        raise ValueError(message)


def read(name):
    return json.loads((BASE/name).read_text())


def verify():
    from .archives import verify as verify_archives
    receipt = read('release_receipt.json')
    require(receipt['qualified_primary'] is None and receipt['deployed'] is False,'Unexpected promotion')
    for name,expected in receipt['file_sha256'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts,'Unsafe receipt path')
        require((BASE/path).is_file() and sha(BASE/path)==expected,'Research artifact changed: '+name)
    archive_result = verify_archives()
    from .cli import load_profile
    for role in read('profiles.json')['variants']:
        load_profile(role)
    return dict(passed=True,files=len(receipt['file_sha256']),archives=archive_result,
                qualified_primary=None,deployed=False)


def seal():
    from .archives import verify as verify_archives
    target = BASE/'release_receipt.json'
    require(not target.exists(),'Research release already sealed')
    protected = protect()
    required = ('REPORT.md','RESULTS.md','profiles.json','runtime_manifest.json','path_archives.json',
        'results/pressure/completion.json','results/family_diagnostics/receipt.json',
        'results/cli_smoke.json','results/render_receipt.json','results/test_receipt.json',
        'results/sensitivity/evaluation.json','results/reselection/evaluation.json')
    for name in required:
        require((BASE/name).is_file(),'Incomplete research evidence: '+name)
    for stage in ('main','consensus','exante'):
        choice = read('results/'+stage+'/selection.json')
        require(choice['primary'] is None and choice['qualified_count']==0,'Stage qualification changed')
        require((BASE/'results/audit_stages'/stage/'audit.json').is_file(),'Missing independent audit')
    archive_result = verify_archives()
    manifest = read('path_archives.json')
    raw_paths = {row['path'] for row in manifest['archives']}
    files = {}
    for path in BASE.rglob('*'):
        relative = path.relative_to(BASE)
        if (not path.is_file() or any(part in ('cache','__pycache__') for part in relative.parts)
                or path.suffix in ('.pyc','.so') or str(relative) in raw_paths or path==target):
            continue
        files[str(relative)] = sha(path)
    dump(target,dict(schema=1,sealed_at=stamp(),file_sha256=files,archives_verified=archive_result,
        prior_files_unchanged=protected,qualified_primary=None,deployed=False,
        status='Completed finite research round; partial historical improvements, generalization unproven'))
    return verify()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal',action='store_true')
    args=parser.parse_args()
    print(json.dumps(seal() if args.seal else verify(),ensure_ascii=False,indent=2))
