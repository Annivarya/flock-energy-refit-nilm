"""Download official REFIT files and extract with SHA256 provenance."""
import hashlib
import json
import urllib.request
from pathlib import Path
import py7zr

ROOT=Path(__file__).resolve().parent
BASE='https://pure.strath.ac.uk/ws/portalfiles/portal/'
FILES={'raw.7z':'62090177/REFIT_RAW_081116.7z', 'cleaned.7z':'62090184/CLEAN_REFIT_081116.7z',
       'RAW_READ_ME.txt':'62090176/RAW_READ_ME_081116.txt', 'CLEAN_READ_ME.txt':'62090183/CLEAN_READ_ME_081116.txt'}

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    folder=ROOT/'data'/'downloads';folder.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for name,suffix in FILES.items():
        path=folder/name
        if not path.exists():
            temp=path.with_suffix(path.suffix+'.partial')
            print('Downloading',name,flush=True)
            request=urllib.request.Request(BASE+suffix,headers={'User-Agent':'REFIT-research/1.0'})
            with urllib.request.urlopen(request,timeout=120) as response,temp.open('wb') as out:
                while chunk:=response.read(1024*1024):out.write(chunk)
            temp.replace(path)
        digest=sha256(path)
        expected_path=ROOT/'reports'/'source_manifest.json'
        if expected_path.exists():
            expected={r['file']:r['sha256'] for r in json.loads(expected_path.read_text())}
            if name in expected and expected[name]!=digest:
                raise ValueError(f'Source checksum changed for {name}; inspect before proceeding')
        manifest.append({'file':name,'url':BASE+suffix,'bytes':path.stat().st_size,'sha256':digest})
    for kind in ['raw','cleaned']:
        target=ROOT/'data'/kind;target.mkdir(exist_ok=True)
        with py7zr.SevenZipFile(folder/f'{kind}.7z') as archive:
            names=archive.getnames()
            targets=names if kind=='cleaned' else [n for n in names if 'House1_' in n]
            if not all((target/n).exists() for n in targets):archive.extract(path=target,targets=targets)
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports'/'source_manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
