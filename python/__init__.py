try:
    from pkg_resources import get_distribution
    __version__ = get_distribution('irbasis-utility').version
except:
    __version__ = '0.0.0'
