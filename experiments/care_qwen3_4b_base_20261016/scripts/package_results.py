import pathlib,zipfile
root=pathlib.Path('.')
dest=pathlib.Path('/home/gcx/CARE_results_20261016.zip')
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
    for f in sorted(root.rglob('*')):
        if not f.is_file(): continue
        parts=f.parts
        if parts[0]=='models' or '__pycache__' in parts or '.pytest_cache' in parts: continue
        if len(parts)>1 and parts[:2]==('figures','templates'): continue
        z.write(f,arcname='CARE_results_20261016/'+f.as_posix())
print(dest, dest.stat().st_size)
