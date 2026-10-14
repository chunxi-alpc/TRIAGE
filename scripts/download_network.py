"""Process-local DNS caching for unreliable research-server DNS.
URLs, HTTPS certificates/SNI, and system DNS settings remain unchanged.
"""
import socket
import threading

# Resolved on this server on 2026-10-14; CAS .62/.69 successfully served ranges.
_SEEDS = {
    "cas-bridge.xethub.hf.co": ["3.168.86.62", "3.168.86.69"],
    "cdn-lfs-cn-1.modelscope.cn": ["42.48.120.198", "42.48.120.197", "42.48.120.202", "42.48.120.195"],
    "modelscope.cn": ["47.92.141.220", "39.99.133.195"],
    "hf-mirror.com": ["160.16.86.14"],
    "api.github.com": ["140.82.114.6"],
    "raw.githubusercontent.com": ["185.199.109.133", "185.199.108.133", "185.199.110.133", "185.199.111.133"],
}

def install_resolution_cache():
    original = socket.getaddrinfo
    cache = dict(_SEEDS)
    lock = threading.Lock()
    def cached(host, port, *args, **kwargs):
        if not isinstance(host, str) or not (host.endswith("modelscope.cn") or host in _SEEDS):
            return original(host, port, *args, **kwargs)
        with lock:
            addresses = cache.get(host)
            if addresses is None:
                answers = original(host, port, *args, **kwargs)
                addresses = list(dict.fromkeys(a[4][0] for a in answers if a[0] == socket.AF_INET))
                if not addresses:
                    return answers
                cache[host] = addresses
        return [answer for ip in addresses for answer in original(ip, port, *args, **kwargs)]
    socket.getaddrinfo = cached
