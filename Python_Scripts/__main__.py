"""
James Smith.

Post Summer Update:

Splitting all files into directory-based module, in order to streamline code,
while making improvements to loading system, label system, etc.

Most-likely will update README.md when most decisions are made on what the final
outcome (at least realistically) will be. (08/18/2018)

~ __main__.py ~

Starting Execution, checking logging/parser arguments.
"""
from .config.cfg import args, cfgLog

if __name__ == '__main__':
    logger = cfgLog.create_logger(__name__)
    logger.info('AHH IM DYING')
    print('It worked', args)
