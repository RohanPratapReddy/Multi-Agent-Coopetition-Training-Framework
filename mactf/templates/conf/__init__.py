"""Master copies of the ``conf/*.yaml`` files that ``startproject`` writes.

Every file ends in ``.tpl`` so it is never imported or parsed in place. The file
list is :data:`mactf.conf_files.CONF_LAYOUT`; a template missing from here is
refused by ``startproject`` and by the packaging tests.
"""
