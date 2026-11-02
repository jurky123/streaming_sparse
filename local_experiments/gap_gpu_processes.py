"""Read-only NVIDIA process inventory, including graphics contexts."""
import subprocess
import xml.etree.ElementTree as ET

def parse_inventory(xml):
    root = ET.fromstring(xml)
    rows = {}
    for gpu in root.findall('gpu'):
        uuid = gpu.findtext('uuid')
        if not uuid:
            raise ValueError('GPU UUID missing in nvidia-smi XML')
        processes = []
        for info in gpu.findall('./processes/process_info'):
            pid = info.findtext('pid')
            if pid is None or not pid.strip().isdigit():
                raise ValueError('GPU process PID unavailable')
            processes.append(dict(pid=int(pid), type=info.findtext('type', 'unknown')))
        if gpu.find('processes') is None:
            raise ValueError('GPU process inventory unavailable')
        rows[uuid] = processes
    if not rows:
        raise ValueError('No GPU inventory returned')
    return rows

def inventory(device=None):
    args = ['nvidia-smi']
    if device is not None:
        args += ['-i', str(device)]
    return parse_inventory(subprocess.check_output(args + ['-q', '-x'], text=True, timeout=20))
