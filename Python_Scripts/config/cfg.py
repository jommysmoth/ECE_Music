"""
James Smith.

Post Summer Update:

Splitting all files into directory-based module, in order to streamline code,
while making improvements to loading system, label system, etc.

Most-likely will update README.md when most decisions are made on what the final
outcome (at least realistically) will be. (08/18/2018).

~ config/cfg.py ~

Incorporate logging and argparse from lower level, for information useful in every part of
development

"""
import argparse
import logging

logger = logging.basicConfig()
parser = argparse.ArgumentParser()

parser.add_argument('-files',
					'--MUSIC_PATH',
					nargs='?',
					default='My/default/path')
args = parser.parse_args()
print(args)
cfglogger = logger
